import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Text, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from mope.config import settings

Base = declarative_base()

# ==========================================
# Local MOPE Database Models (mope.db)
# ==========================================

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(64), nullable=False)  # 'burnout_xgb', 'anxiety_xgb', 'depressive_lgb'
    version = Column(String(32), nullable=False)
    file_path = Column(String(256), nullable=False)
    metric_auc = Column(Float, nullable=True)
    metric_f1 = Column(Float, nullable=True)
    metric_mae = Column(Float, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint('model_type', 'version', name='_model_type_version_uc'),
    )


class MopePrediction(Base):
    __tablename__ = "mope_predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(64), nullable=False, index=True)
    prediction_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    burnout_probability = Column(Float, nullable=False)
    anxiety_score = Column(Float, nullable=False)
    anxiety_level_risk = Column(String(16), nullable=False)  # 'Low', 'Medium', 'High'
    depressive_onset_index = Column(Float, nullable=False)
    critical_threshold_breached = Column(Boolean, default=False, nullable=False)
    anomaly_warning = Column(Boolean, default=False, nullable=False)
    model_version = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ==========================================
# External SDT Database Models (sdt.db)
# ==========================================

# We map these read-only models to access the Digital Twin state and history
SdtBase = declarative_base()

class SdtDigitalTwinState(SdtBase):
    __tablename__ = "digital_twin_states"
    
    student_id = Column(String(64), primary_key=True)
    s_stress = Column(Float, nullable=False)
    s_anxiety = Column(Float, nullable=False)
    s_fatigue = Column(Float, nullable=False)
    s_social = Column(Float, nullable=False)
    s_academic = Column(Float, nullable=False)
    s_burnout = Column(Float, nullable=False)
    s_sleep = Column(Float, nullable=False)
    s_mood = Column(Float, nullable=False)
    s_resilience = Column(Float, nullable=False)
    s_focus = Column(Float, nullable=False)
    last_update_epoch = Column(Integer, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class SdtTwinStateHistory(SdtBase):
    __tablename__ = "twin_state_history"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    encrypted_payload = Column(Text, nullable=False)
    key_id = Column(String(64), nullable=False)
    trigger_source = Column(String(128), nullable=False)


class SdtEncryptionKey(SdtBase):
    __tablename__ = "encryption_keys"
    
    id = Column(String(64), primary_key=True)
    key_bytes = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, nullable=False)
    active = Column(Boolean, nullable=False)


# ==========================================
# Database Connection Setup
# ==========================================

# 1. Local Database (mope.db)
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 2. SDT Database (sdt.db)
sdt_engine = create_engine(
    settings.SDT_DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.SDT_DATABASE_URL else {}
)
SdtSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sdt_engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_sdt_db():
    db = SdtSessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Ensure folders exist
    os.makedirs(settings.MODEL_REGISTRY_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
    Base.metadata.create_all(bind=engine)
