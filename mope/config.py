import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Outcome Prediction Engine (MOPE)"
    # Base database for predictions and registry
    DATABASE_URL: str = "sqlite:///mope.db"
    # Reference database to load student digital twin states
    SDT_DATABASE_URL: str = "sqlite:///../Digital Twin/sdt.db"
    PORT: int = 8002
    HOST: str = "0.0.0.0"
    REDIS_URL: str = "redis://localhost:6379/0"
    MODEL_REGISTRY_DIR: str = "data/models"
    
    # Alert thresholds
    BURNOUT_THRESHOLD: float = 0.70
    ANXIETY_THRESHOLD: float = 0.70
    DEPRESSIVE_THRESHOLD: float = 0.60
    
    # Anomaly trigger jump thresholds
    BURNOUT_JUMP_THRESHOLD: float = 0.30
    DEPRESSIVE_JUMP_THRESHOLD: float = 0.25
    
    class Config:
        env_file = ".env"

settings = Settings()

# Resolve absolute paths for SQLite to avoid relative directory issues
def get_absolute_db_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        relative_path = url.replace("sqlite:///", "")
        # If relative, make it absolute from this file's root
        if not os.path.isabs(relative_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_path = os.path.normpath(os.path.join(base_dir, relative_path))
            return f"sqlite:///{abs_path}"
    return url

settings.DATABASE_URL = get_absolute_db_url(settings.DATABASE_URL)
settings.SDT_DATABASE_URL = get_absolute_db_url(settings.SDT_DATABASE_URL)
