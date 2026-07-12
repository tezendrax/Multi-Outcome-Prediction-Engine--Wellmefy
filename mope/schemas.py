from pydantic import BaseModel, Field
from typing import Optional

class TemporalMetadata(BaseModel):
    semester_week: int = Field(default=6, ge=1, le=18, description="Current week of the semester")
    days_to_midterms: Optional[float] = Field(default=None, description="Days remaining until midterms")

class PredictionTriggerRequest(BaseModel):
    student_id: str
    temporal_metadata: Optional[TemporalMetadata] = Field(default_factory=TemporalMetadata)

class PredictionsInner(BaseModel):
    burnout_risk: float = Field(..., description="Probability of burnout [0.0, 1.0]")
    anxiety_score: float = Field(..., description="Normalized anxiety severity score [0.0, 1.0]")
    clinical_indicator_alert: bool = Field(..., description="Flag indicating if any clinical indicators breach safety levels")

class PredictionsDetails(BaseModel):
    burnout_probability: float = Field(..., description="Burnout risk probability")
    anxiety_level_risk: str = Field(..., description="Categorical anxiety risk level (Low, Medium, High)")
    depressive_onset_index: float = Field(..., description="Predicted depressive onset index [0.0, 1.0]")
    critical_threshold_breached: bool = Field(..., description="Indicates if any baseline safety threshold is breached")
    anomaly_warning: bool = Field(..., description="Signals abnormal short-term changes in predictions")

class PredictionsResponse(BaseModel):
    student_id: str = Field(..., description="The unique student ID")
    predictions: PredictionsInner = Field(..., description="Summary predictions conforming to upstream intervention schema")
    details: PredictionsDetails = Field(..., description="Detailed prediction diagnostics")
    model_version: str = Field(..., description="Active prediction model registry version")
    timestamp: str = Field(..., description="Prediction creation timestamp in ISO 8601 format")
