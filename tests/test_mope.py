import sys
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mope.config import settings
from mope.db import init_db, SessionLocal, ModelRegistry, MopePrediction
from mope.features import preprocess_and_impute_history, extract_features, DIMENSIONS
from mope.anomaly import detect_anomaly
from mope.main import app
from mope.inference import engine_singleton

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Initializes the database and ensures it has active model mappings for tests."""
    init_db()
    
    # We will register mock model records for testing if they are not already there
    db = SessionLocal()
    try:
        # Check if active models exist, if not, insert dummy ones
        active = db.query(ModelRegistry).filter(ModelRegistry.active == True).first()
        if not active:
            dummy_models = [
                ModelRegistry(
                    model_type="burnout_xgb",
                    version="v1.4.2",
                    file_path=os.path.join(settings.MODEL_REGISTRY_DIR, "burnout_xgb_v1.4.2.joblib"),
                    metric_auc=0.98,
                    metric_f1=0.91,
                    active=True
                ),
                ModelRegistry(
                    model_type="anxiety_xgb",
                    version="v1.4.2",
                    file_path=os.path.join(settings.MODEL_REGISTRY_DIR, "anxiety_xgb_v1.4.2.joblib"),
                    metric_f1=0.77,
                    active=True
                ),
                ModelRegistry(
                    model_type="depressive_lgb",
                    version="v1.4.2",
                    file_path=os.path.join(settings.MODEL_REGISTRY_DIR, "depressive_lgb_v1.4.2.joblib"),
                    metric_mae=0.03,
                    active=True
                )
            ]
            db.add_all(dummy_models)
            db.commit()
    finally:
        db.close()
    yield

def test_preprocess_and_impute_history():
    """Tests that daily resampling and LOCF imputation work as expected."""
    # 1. Test empty history (should return baseline)
    df_empty = preprocess_and_impute_history([], num_days=14)
    assert len(df_empty) == 14
    assert list(df_empty.columns) == DIMENSIONS
    assert df_empty.loc[df_empty.index[0], "stress"] == 0.30
    
    # 2. Test history with gaps
    base_time = datetime.utcnow()
    records = [
        {"timestamp": base_time - timedelta(days=5), "stress": 0.4, "sleep": 0.8},
        {"timestamp": base_time - timedelta(days=2), "stress": 0.6, "sleep": 0.5},
        {"timestamp": base_time, "stress": 0.7, "sleep": 0.6}
    ]
    # fill other dimensions
    for r in records:
        for dim in DIMENSIONS:
            if dim not in r:
                r[dim] = 0.5
                
    df = preprocess_and_impute_history(records, num_days=7)
    assert len(df) == 7
    # The last date should be base_time.date()
    assert df.index[-1] == base_time.date()
    # Stress on the last day should be 0.7
    assert df.iloc[-1]["stress"] == 0.7
    # Day -1 should be 0.6 (forward filled from day -2)
    assert df.iloc[-2]["stress"] == 0.6
    # Day -3 should be 0.4 (forward filled from day -5)
    assert df.iloc[-4]["stress"] == 0.4

def test_extract_features():
    """Tests derived metrics and rolling window equations."""
    dates = [datetime.utcnow().date() - timedelta(days=i) for i in range(14)]
    dates.reverse()
    
    # Generate 14 days of states
    data = {dim: [0.5] * 14 for dim in DIMENSIONS}
    # Induce stress increase
    data["stress"] = [0.2] * 7 + [0.6] * 7 # stress jump
    data["sleep"] = [0.8] * 14
    
    df = pd.DataFrame(data, index=dates)
    features = extract_features(df, days_to_midterms=10.0)
    
    assert features["latest_stress"] == 0.6
    assert features["delta_stress_7d"] == pytest.approx(0.4)
    assert features["sleep_stress_ratio"] == pytest.approx(0.8 / 0.6, 1e-4)
    assert features["days_to_midterms"] == 10.0
    assert features["mean_stress_7d"] == pytest.approx(0.6)
    assert features["mean_stress_14d"] == pytest.approx(0.4)
    assert "social_volatility_14d" in features
    assert "trend_stress_7d" in features

def test_anomaly_detection():
    """Tests that Anomaly Detector raises flags for spikes."""
    db = SessionLocal()
    student_id = "test-anomaly-student"
    
    # Clear prior test predictions
    db.query(MopePrediction).filter(MopePrediction.student_id == student_id).delete()
    db.commit()
    
    # 1. No history: should be False
    is_anomaly = detect_anomaly(db, student_id, 0.5, "Low", 0.3)
    assert is_anomaly is False
    
    # Write a baseline prediction run
    baseline_pred = MopePrediction(
        student_id=student_id,
        burnout_probability=0.30,
        anxiety_score=0.40,
        anxiety_level_risk="Low",
        depressive_onset_index=0.20,
        model_version="v1.4.2"
    )
    db.add(baseline_pred)
    db.commit()
    
    # 2. Under threshold increases: should be False
    is_anomaly = detect_anomaly(db, student_id, 0.45, "Medium", 0.30)
    assert is_anomaly is False
    
    # 3. Burnout jump > 0.30: should trigger True
    is_anomaly = detect_anomaly(db, student_id, 0.65, "Low", 0.20)
    assert is_anomaly is True
    
    # 4. Depressive onset jump > 0.25: should trigger True
    is_anomaly = detect_anomaly(db, student_id, 0.35, "Low", 0.50)
    assert is_anomaly is True
    
    # 5. Anxiety transition Low -> High: should trigger True
    is_anomaly = detect_anomaly(db, student_id, 0.30, "High", 0.20)
    assert is_anomaly is True
    
    db.close()

def test_api_health_check():
    """Tests that health check returns status of services."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "registered_models" in data

def test_api_get_predictions_fallback():
    """Tests predicting for a mock student (uses baseline state fallback)."""
    # Using a fake student who has no entry in sdt.db should fall back to default
    response = client.get("/api/v1/predictions/mope?student_id=fake-student-007")
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "fake-student-007"
    assert "predictions" in data
    assert "burnout_risk" in data["predictions"]
    assert "anxiety_score" in data["predictions"]
    assert "clinical_indicator_alert" in data["predictions"]
    assert "details" in data
    assert "model_version" in data
