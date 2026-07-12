import os
import time
import json
import logging
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy.orm import Session
from mope.config import settings
from mope.db import SessionLocal, get_sdt_db, ModelRegistry, MopePrediction, SdtDigitalTwinState, SdtTwinStateHistory
from mope.features import decrypt_history_records, preprocess_and_impute_history, extract_features, DIMENSIONS
from mope.anomaly import detect_anomaly

logger = logging.getLogger("mope_inference")

# Initialize Redis client (with fallback to in-memory cache)
redis_client = None
try:
    import redis
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
    redis_client.ping()
    logger.info("Connected to Redis successfully for MOPE caching.")
except Exception as e:
    logger.warning(f"Redis connection failed ({e}). Falling back to in-memory prediction cache.")
    redis_client = None

# In-memory backup cache
_in_memory_cache = {}

def get_cache(student_id: str) -> dict:
    """Retrieves cached prediction results for a student if valid (< 1 hour)."""
    now = time.time()
    
    # 1. Try Redis first
    if redis_client:
        try:
            cached_data = redis_client.get(f"mope:pred:{student_id}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Redis cache get failed: {e}")
            
    # 2. Try In-memory cache fallback
    if student_id in _in_memory_cache:
        entry = _in_memory_cache[student_id]
        if now - entry["cached_at"] < 3600:  # 1 hour expiry
            return entry["data"]
        else:
            del _in_memory_cache[student_id]
            
    return None

def set_cache(student_id: str, data: dict):
    """Caches prediction results for a student for 1 hour."""
    now = time.time()
    
    # 1. Try Redis first
    if redis_client:
        try:
            redis_client.setex(
                f"mope:pred:{student_id}",
                3600,  # 1 hour in seconds
                json.dumps(data)
            )
            return
        except Exception as e:
            logger.warning(f"Redis cache set failed: {e}")
            
    # 2. Try In-memory cache fallback
    _in_memory_cache[student_id] = {
        "cached_at": now,
        "data": data
    }

class InferenceEngine:
    def __init__(self):
        self.active_version = None
        self.burnout_model = None
        self.anxiety_model = None
        self.depressive_model = None
        self.features_list = None
        
    def load_active_models(self, db: Session):
        """Loads models marked active in the database registry."""
        # Query model registry for active models
        registry_entries = db.query(ModelRegistry).filter(ModelRegistry.active == True).all()
        if not registry_entries:
            raise RuntimeError("No active models registered in the database registry. Please run model training first.")
            
        version = registry_entries[0].version
        
        # If already loaded the same version, skip
        if self.active_version == version and self.burnout_model is not None:
            return
            
        # Get individual model files
        burnout_entry = next((r for r in registry_entries if r.model_type == "burnout_xgb"), None)
        anxiety_entry = next((r for r in registry_entries if r.model_type == "anxiety_xgb"), None)
        depressive_entry = next((r for r in registry_entries if r.model_type == "depressive_lgb"), None)
        
        if not (burnout_entry and anxiety_entry and depressive_entry):
            raise RuntimeError(f"Active model registry for version {version} is incomplete.")
            
        # Load files
        try:
            self.burnout_model = joblib.load(burnout_entry.file_path)
            self.anxiety_model = joblib.load(anxiety_entry.file_path)
            self.depressive_model = joblib.load(depressive_entry.file_path)
            
            # Load feature column names list
            features_path = os.path.join(settings.MODEL_REGISTRY_DIR, f"features_list_{version}.joblib")
            if os.path.exists(features_path):
                self.features_list = joblib.load(features_path)
            else:
                # Fallback to reconstructing list
                from mope.features import get_feature_names
                self.features_list = get_feature_names()
                
            self.active_version = version
            logger.info(f"Successfully loaded MOPE active models version: {version}")
        except Exception as e:
            raise RuntimeError(f"Error loading model binary files for version {version}: {e}")

    def run_prediction(
        self, 
        student_id: str, 
        days_to_midterms: float = 14.0, 
        force_recompute: bool = False
    ) -> dict:
        """
        Runs the full prediction pipeline for a student.
        1. Checks cache first.
        2. Retrieves last 14 days of states from sdt.db.
        3. Preprocesses, imputes, and extracts temporal feature vectors.
        4. Runs inference on models.
        5. Performs Anomaly Detection against historical prediction runs.
        6. Standardizes predictions and saves to mope_predictions.
        7. Caches the result.
        """
        # Check cache if not forced
        if not force_recompute:
            cached = get_cache(student_id)
            if cached:
                return cached

        db = SessionLocal()
        sdt_db = next(get_sdt_db())
        
        try:
            # 1. Ensure models are loaded
            self.load_active_models(db)
            
            # 2. Get student history from SDT database
            history_records = sdt_db.query(SdtTwinStateHistory)\
                .filter(SdtTwinStateHistory.student_id == student_id)\
                .order_by(SdtTwinStateHistory.timestamp.desc())\
                .limit(100)\
                .all()
                
            # Reverse history to chronological order
            history_records.reverse()
            
            # Get latest current twin state to use as reference baseline or fallback
            twin_state = sdt_db.query(SdtDigitalTwinState).filter(SdtDigitalTwinState.student_id == student_id).first()
            
            baseline = None
            if twin_state:
                baseline = {dim: getattr(twin_state, f"s_{dim}") for dim in DIMENSIONS}
                
            # 3. Decrypt and preprocess
            decrypted = decrypt_history_records(sdt_db, history_records)
            df = preprocess_and_impute_history(decrypted, num_days=14, baseline_state=baseline)
            
            # 4. Feature Extraction
            feature_dict = extract_features(df, days_to_midterms)
            
            # Create a 1-row DataFrame aligned with features_list
            X = pd.DataFrame([feature_dict])[self.features_list]
            
            # 5. Run Model Inference
            # Burnout Risk probability
            burnout_prob = float(self.burnout_model.predict_proba(X)[:, 1][0])
            
            # Anxiety Level Risk (multiclass)
            anxiety_probs = self.anxiety_model.predict_proba(X)[0]
            anxiety_score = float(anxiety_probs[1] * 0.5 + anxiety_probs[2] * 1.0)
            anxiety_class_idx = int(np.argmax(anxiety_probs))
            anxiety_level = ["Low", "Medium", "High"][anxiety_class_idx]
            
            # Depressive Onset Index (regression)
            depressive_index = float(self.depressive_model.predict(X)[0])
            depressive_index = max(0.0, min(1.0, depressive_index))  # clip bounds
            
            # 6. Anomaly Detection
            anomaly_warn = detect_anomaly(
                db=db,
                student_id=student_id,
                current_burnout=burnout_prob,
                current_anxiety_level=anxiety_level,
                current_depressive=depressive_index
            )
            
            # 7. Check Critical Threshold Breaches
            critical_breached = (
                burnout_prob >= settings.BURNOUT_THRESHOLD or
                anxiety_score >= settings.ANXIETY_THRESHOLD or
                depressive_index >= settings.DEPRESSIVE_THRESHOLD
            )
            
            # Save prediction to DB
            pred_record = MopePrediction(
                student_id=student_id,
                prediction_timestamp=datetime.utcnow(),
                burnout_probability=burnout_prob,
                anxiety_score=anxiety_score,
                anxiety_level_risk=anxiety_level,
                depressive_onset_index=depressive_index,
                critical_threshold_breached=critical_breached,
                anomaly_warning=anomaly_warn,
                model_version=self.active_version
            )
            db.add(pred_record)
            db.commit()
            db.refresh(pred_record)
            
            # 8. Standardize Output Schema for Response
            response_payload = {
                "student_id": student_id,
                "predictions": {
                    "burnout_risk": round(burnout_prob, 4),
                    "anxiety_score": round(anxiety_score, 4),
                    "clinical_indicator_alert": bool(critical_breached or anomaly_warn)
                },
                "details": {
                    "burnout_probability": round(burnout_prob, 4),
                    "anxiety_level_risk": anxiety_level,
                    "depressive_onset_index": round(depressive_index, 4),
                    "critical_threshold_breached": bool(critical_breached),
                    "anomaly_warning": bool(anomaly_warn)
                },
                "model_version": self.active_version,
                "timestamp": pred_record.prediction_timestamp.isoformat() + "Z"
            }
            
            # Cache results
            set_cache(student_id, response_payload)
            
            return response_payload
            
        except Exception as e:
            db.rollback()
            logger.exception(f"Inference error for student {student_id}: {e}")
            raise
        finally:
            db.close()
            sdt_db.close()

# Singleton engine instance
engine_singleton = InferenceEngine()
