import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session
from mope.db import SdtTwinStateHistory, SdtEncryptionKey

DIMENSIONS = ["stress", "anxiety", "fatigue", "social", "academic", "burnout", "sleep", "mood", "resilience", "focus"]

def decrypt_history_records(db_sdt: Session, records: List[SdtTwinStateHistory]) -> List[Dict[str, Any]]:
    """Decrypts history payloads using the respective key_ids from the database."""
    decrypted_records = []
    # Cache keys to avoid querying the DB repeatedly for the same key_id
    key_cache = {}
    
    for rec in records:
        key_id = rec.key_id
        if key_id not in key_cache:
            key_rec = db_sdt.query(SdtEncryptionKey).filter(SdtEncryptionKey.id == key_id).first()
            if key_rec:
                key_cache[key_id] = Fernet(key_rec.key_bytes)
            else:
                continue  # Skip if key not found
                
        f = key_cache[key_id]
        try:
            decrypted_bytes = f.decrypt(rec.encrypted_payload.encode('utf-8'))
            state_dict = pd.read_json(decrypted_bytes.decode('utf-8'), typ='series').to_dict()
            state_dict['timestamp'] = rec.timestamp
            decrypted_records.append(state_dict)
        except Exception:
            # Skip records that fail decryption (corrupted or wrong key)
            continue
            
    return decrypted_records

def preprocess_and_impute_history(
    decrypted_records: List[Dict[str, Any]], 
    num_days: int = 14, 
    baseline_state: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Groups, resamples and imputes student twin state history.
    1. Extracts state vectors and timestamps.
    2. Takes the last observation of each calendar day.
    3. Fills gaps using Last-Observation-Carried-Forward (LOCF).
    4. Backfills with baseline values if the history is shorter than num_days.
    """
    if not baseline_state:
        # Default baseline values if none provided
        baseline_state = {
            "stress": 0.3, "anxiety": 0.25, "fatigue": 0.3, "social": 0.7,
            "academic": 0.6, "burnout": 0.15, "sleep": 0.75, "mood": 0.7,
            "resilience": 0.65, "focus": 0.7
        }
        
    if not decrypted_records:
        # No history: Return static dataframe of baseline values
        dates = [datetime.utcnow().date() - timedelta(days=i) for i in range(num_days)]
        dates.reverse()
        df = pd.DataFrame([baseline_state] * num_days)
        df['date'] = dates
        return df.set_index('date')

    # Convert to DataFrame
    df = pd.DataFrame(decrypted_records)
    df['date'] = df['timestamp'].dt.date
    
    # Take latest state for each day (LOCF for intraday)
    df = df.sort_values('timestamp').groupby('date').last()
    df = df[DIMENSIONS]  # Keep only the 10 state dimensions
    
    # Reindex to a complete daily range ending at the latest date
    latest_date = max(df.index)
    start_date = latest_date - timedelta(days=num_days - 1)
    full_date_range = pd.date_range(start=start_date, end=latest_date).date
    
    # Reindex
    df = df.reindex(full_date_range)
    
    # Impute missing values using LOCF (forward fill)
    df = df.ffill()
    
    # If the first values are NaN (because history started after start_date), backfill with baseline
    for col in DIMENSIONS:
        df[col] = df[col].fillna(baseline_state[col])
        
    return df

def extract_features(df: pd.DataFrame, days_to_midterms: float) -> Dict[str, float]:
    """
    Calculates advanced tabular features from a 14-day resampled history DataFrame.
    Returns a dictionary of features mapping to model inputs.
    """
    features = {}
    
    # Ensure index is sorted chronologically
    df = df.sort_index()
    
    # 1. Latest states (t)
    latest = df.iloc[-1]
    for dim in DIMENSIONS:
        features[f"latest_{dim}"] = float(latest[dim])
        
    # 2. Historical states (t-7)
    state_7d_ago = df.iloc[-8] if len(df) >= 8 else df.iloc[0]
    
    # 3. Derived Metrics
    # Delta Stress (t) - (t-7)
    features["delta_stress_7d"] = float(latest["stress"] - state_7d_ago["stress"])
    
    # Sleep-to-Stress Ratio
    features["sleep_stress_ratio"] = float(latest["sleep"] / (latest["stress"] + 1e-5))
    
    # Social Interaction Volatility (14-day rolling standard deviation)
    features["social_volatility_14d"] = float(df["social"].std(ddof=0))
    
    # Academic temporal feature
    features["days_to_midterms"] = float(days_to_midterms)
    
    # 4. Rolling Averages and Standard Deviations
    for dim in DIMENSIONS:
        # 7-day stats
        sub_7 = df.iloc[-7:]
        features[f"mean_{dim}_7d"] = float(sub_7[dim].mean())
        features[f"std_{dim}_7d"] = float(sub_7[dim].std(ddof=0))
        
        # 14-day stats
        features[f"mean_{dim}_14d"] = float(df[dim].mean())
        features[f"std_{dim}_14d"] = float(df[dim].std(ddof=0))
        
        # Gradients (slope of linear regression on the last 7 days to represent trend)
        x = np.arange(len(sub_7))
        y = sub_7[dim].values
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0
        features[f"trend_{dim}_7d"] = slope
        
    return features

def get_feature_names() -> List[str]:
    """Helper to return the exact list of feature names used by models."""
    # We will generate a dummy dataframe and run feature extraction to get names
    dummy_dates = [datetime.utcnow().date() - timedelta(days=i) for i in range(14)]
    dummy_dates.reverse()
    dummy_data = {dim: [0.5] * 14 for dim in DIMENSIONS}
    df = pd.DataFrame(dummy_data, index=dummy_dates)
    features = extract_features(df, 14.0)
    return list(features.keys())
