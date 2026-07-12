from sqlalchemy.orm import Session
from mope.db import MopePrediction
from mope.config import settings

def detect_anomaly(
    db: Session,
    student_id: str,
    current_burnout: float,
    current_anxiety_level: str,
    current_depressive: float
) -> bool:
    """
    Compares current predictions with the student's last recorded prediction.
    Triggers an anomaly warning flag (True) if:
    1. Burnout risk increases by > 0.30 (BURNOUT_JUMP_THRESHOLD)
    2. Depressive onset index increases by > 0.25 (DEPRESSIVE_JUMP_THRESHOLD)
    3. Anxiety level jumps directly from 'Low' to 'High'
    """
    # Retrieve the last prediction for this student
    last_pred = db.query(MopePrediction)\
        .filter(MopePrediction.student_id == student_id)\
        .order_by(MopePrediction.prediction_timestamp.desc())\
        .first()
        
    if not last_pred:
        return False  # No historical predictions to compare with, so no anomaly warning yet
        
    # Check Burnout jump
    burnout_jump = current_burnout - last_pred.burnout_probability
    if burnout_jump >= settings.BURNOUT_JUMP_THRESHOLD:
        return True
        
    # Check Depressive Index jump
    depressive_jump = current_depressive - last_pred.depressive_onset_index
    if depressive_jump >= settings.DEPRESSIVE_JUMP_THRESHOLD:
        return True
        
    # Check Anxiety transition (Low -> High)
    if last_pred.anxiety_level_risk == "Low" and current_anxiety_level == "High":
        return True
        
    return False
