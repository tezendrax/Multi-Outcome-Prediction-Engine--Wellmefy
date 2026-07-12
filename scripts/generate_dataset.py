import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

# Ensure reproducible datasets
np.random.seed(42)

DIMENSIONS = ["stress", "anxiety", "fatigue", "social", "academic", "burnout", "sleep", "mood", "resilience", "focus"]

def try_load_huggingface_baselines() -> Dict[str, Any]:
    """
    Attempts to load the '0xmarvel/student-stress-survey' dataset from Hugging Face
    to extract statistical profiles (mean and std of sleep, stress, etc.).
    If offline or unavailable, returns realistic default baseline statistics.
    """
    try:
        from datasets import load_dataset
        print("Loading baseline distributions from Hugging Face (0xmarvel/student-stress-survey)...")
        dataset = load_dataset("0xmarvel/student-stress-survey", split="train", trust_remote_code=True)
        df = dataset.to_pandas()
        
        # Extract sleep and stress distributions from the dataset if they exist
        # Typically surveys contain variables like sleep_hours (or quality) and stress_level
        # Let's map typical columns if they exist
        print(f"Loaded survey dataset with {len(df)} rows and columns: {list(df.columns)}")
        
        # We compute some statistics to guide our generator
        stats = {
            "mean_sleep": float(df.get("Sleep Duration", df.get("sleep_hours", pd.Series([7.0]))).mean()) / 8.0,
            "mean_stress": float(df.get("Stress Level", df.get("stress", pd.Series([3.0]))).mean()) / 5.0,
            "hf_available": True
        }
        print(f"Extracted baseline statistics from HF: {stats}")
        return stats
    except Exception as e:
        print(f"Could not load Hugging Face survey dataset ({e}). Using robust default physiological baselines.")
        return {
            "mean_sleep": 0.75,  # ~6 hours
            "mean_stress": 0.35,
            "hf_available": False
        }

def simulate_student_history(
    student_id: str, 
    num_days: int = 20, 
    hf_baselines: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Simulates a daily student state history over a given semester timeline.
    Injects realistic correlations (e.g., sleep loss increases fatigue and stress,
    academic midterms create academic stress, resilience buffers burnout).
    """
    if not hf_baselines:
        hf_baselines = {"mean_sleep": 0.75, "mean_stress": 0.35}
        
    # Baseline states for this student (with some person-level random intercepts)
    person_variance = np.random.normal(0, 0.08, len(DIMENSIONS))
    person_variance = np.clip(person_variance, -0.15, 0.15)
    
    # Base states
    baselines = {
        "stress": max(0.1, min(0.9, hf_baselines["mean_stress"] + person_variance[0])),
        "anxiety": max(0.1, min(0.9, 0.25 + person_variance[1])),
        "fatigue": max(0.1, min(0.9, 0.30 + person_variance[2])),
        "social": max(0.1, min(0.9, 0.70 + person_variance[3])),
        "academic": max(0.1, min(0.9, 0.60 + person_variance[4])),
        "burnout": max(0.05, min(0.9, 0.15 + person_variance[5])),
        "sleep": max(0.1, min(0.9, hf_baselines["mean_sleep"] + person_variance[6])),
        "mood": max(0.1, min(0.9, 0.70 + person_variance[7])),
        "resilience": max(0.2, min(0.95, 0.65 + person_variance[8])),
        "focus": max(0.1, min(0.9, 0.70 + person_variance[9]))
    }
    
    # Track daily states
    daily_records = []
    current_state = baselines.copy()
    
    # Pick a random semester start week
    start_week = np.random.randint(1, 10)
    
    # Timeline
    for day in range(num_days):
        current_week = start_week + (day // 7)
        # Midterms peak at week 8. Calculate days remaining.
        days_to_midterms = max(0.0, (8.0 - (current_week + (day % 7) / 7.0)) * 7.0)
        
        # Academic load multiplier increases as midterms get closer
        academic_load = 1.0
        if days_to_midterms <= 14:
            academic_load = 1.0 + (14 - days_to_midterms) * 0.1  # increases up to 2.4x
            
        # Ingest daily random shocks
        shocks = np.random.normal(0, 0.05, len(DIMENSIONS))
        
        # State transitions with cross-dimension causal equations
        # 1. Academic load drives academic stress and decays sleep
        current_state["academic"] = np.clip(baselines["academic"] - 0.08 * academic_load + shocks[4], 0.0, 1.0)
        current_state["sleep"] = np.clip(baselines["sleep"] - 0.05 * academic_load + shocks[6], 0.0, 1.0)
        
        # 2. Sleep loss drives fatigue and stress
        sleep_loss = max(0.0, 0.75 - current_state["sleep"])
        current_state["fatigue"] = np.clip(baselines["fatigue"] + 0.3 * sleep_loss + shocks[2], 0.0, 1.0)
        current_state["stress"] = np.clip(baselines["stress"] + 0.25 * sleep_loss + 0.1 * (2.0 - current_state["academic"]) + shocks[0], 0.0, 1.0)
        
        # 3. Stress drives anxiety and decays mood
        current_state["anxiety"] = np.clip(baselines["anxiety"] + 0.3 * current_state["stress"] - 0.1 * current_state["social"] + shocks[1], 0.0, 1.0)
        current_state["mood"] = np.clip(baselines["mood"] - 0.3 * current_state["stress"] + 0.1 * current_state["social"] + shocks[7], 0.0, 1.0)
        
        # 4. Social interaction drops under high academic load but helps mood
        current_state["social"] = np.clip(baselines["social"] - 0.05 * academic_load + shocks[3], 0.0, 1.0)
        
        # 5. Burnout accumulates if stress and fatigue are high, buffered by resilience
        burnout_accumulation = max(0.0, (current_state["stress"] + current_state["fatigue"]) / 2.0 - 0.5 * current_state["resilience"])
        current_state["burnout"] = np.clip(current_state["burnout"] + 0.15 * burnout_accumulation + shocks[5], 0.0, 1.0)
        
        # 6. Resilience decays slightly under prolonged burnout but baseline holds
        current_state["resilience"] = np.clip(baselines["resilience"] - 0.1 * current_state["burnout"] + shocks[8], 0.0, 1.0)
        
        # 7. Focus decays with fatigue
        current_state["focus"] = np.clip(baselines["focus"] - 0.3 * current_state["fatigue"] + shocks[9], 0.0, 1.0)
        
        # Capture record
        rec = current_state.copy()
        rec["student_id"] = student_id
        rec["day"] = day
        rec["days_to_midterms"] = days_to_midterms
        daily_records.append(rec)
        
    return pd.DataFrame(daily_records)

def build_tabular_dataset(num_students: int = 800, hf_baselines: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Builds a large tabular dataset of student records.
    For each student, generates 20 days of timeline, computes rolling features,
    and extracts the final day as a training example. This yields 10,000+ raw daily points,
    and we extract the final prediction rows for our training.
    Actually, to get 10k+ rows of final dataset, we can generate more students, or keep multiple
    samples per student (e.g. days 14 to 20 as valid dataset rows).
    If we have 800 students and extract days 14-20 as samples, we get 800 * 7 = 5,600 rows.
    If we have 1,500 students and extract days 14-20, we get 10,500 rows!
    Let's use 1,500 students to ensure a dataset of 10,000+ training/test rows!
    """
    print(f"Simulating daily trajectories for {num_students} students...")
    all_data = []
    
    for i in range(num_students):
        student_id = f"std-{1000 + i}"
        # Simulate 21 days for each student (need at least 14 days for rolling window)
        history_df = simulate_student_history(student_id, num_days=21, hf_baselines=hf_baselines)
        
        # For each student, compute features for days 14 to 20 (7 samples per student)
        # This gives us a dataset of 1,500 * 7 = 10,500 samples!
        for day_idx in range(14, 21):
            sub_df = history_df.iloc[day_idx - 14 : day_idx + 1] # 15 days window
            
            # Extract features using our feature equations
            latest = sub_df.iloc[-1]
            state_7d_ago = sub_df.iloc[-7]
            
            # Feature dictionary
            feats = {}
            feats["student_id"] = latest["student_id"]
            feats["day"] = latest["day"]
            feats["days_to_midterms"] = float(latest["days_to_midterms"])
            
            # 1. Latest states (t)
            for dim in DIMENSIONS:
                feats[f"latest_{dim}"] = float(latest[dim])
                
            # 2. Rolling stats
            for dim in DIMENSIONS:
                sub_7 = sub_df.iloc[-7:]
                feats[f"mean_{dim}_7d"] = float(sub_7[dim].mean())
                feats[f"std_{dim}_7d"] = float(sub_7[dim].std(ddof=0))
                
                feats[f"mean_{dim}_14d"] = float(sub_df[dim].mean())
                feats[f"std_{dim}_14d"] = float(sub_df[dim].std(ddof=0))
                
                # Slope
                x = np.arange(len(sub_7))
                y = sub_7[dim].values
                slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else 0.0
                feats[f"trend_{dim}_7d"] = slope
                
            # Derived Metrics
            feats["delta_stress_7d"] = float(latest["stress"] - state_7d_ago["stress"])
            feats["sleep_stress_ratio"] = float(latest["sleep"] / (latest["stress"] + 1e-5))
            feats["social_volatility_14d"] = float(sub_df["social"].std(ddof=0))
            
            # Define Ground Truth Labels mathematically to mimic clinical evaluations
            # 1. Burnout Probability (Continuous) -> Binary label (>0.55 threshold is burnout)
            # Burnout risk rises with high rolling stress, high rolling fatigue, and low resilience
            burnout_risk = 0.4 * feats["latest_burnout"] + 0.3 * feats["mean_stress_7d"] + 0.2 * feats["mean_fatigue_7d"] - 0.1 * feats["latest_resilience"]
            burnout_risk = max(0.0, min(1.0, burnout_risk + np.random.normal(0, 0.03)))
            feats["burnout_probability_target"] = burnout_risk
            feats["burnout_label"] = 1 if burnout_risk > 0.55 else 0
            
            # 2. Anxiety level risk (categorical: Low, Medium, High)
            # Anxiety score rises with latest anxiety, high stress, and low sleep
            anxiety_score = 0.55 * feats["latest_anxiety"] + 0.35 * feats["mean_stress_7d"] + 0.1 * (1.0 - feats["mean_sleep_7d"])
            anxiety_score = max(0.0, min(1.0, anxiety_score + np.random.normal(0, 0.03)))
            feats["anxiety_score_target"] = anxiety_score
            if anxiety_score < 0.45:
                feats["anxiety_label"] = 0  # Low
            elif anxiety_score < 0.55:
                feats["anxiety_label"] = 1  # Medium
            else:
                feats["anxiety_label"] = 2  # High
                
            # 3. Depressive onset index (float [0.0, 1.0])
            # Depressive risk rises with sustained low mood, low resilience, and low social interaction
            depressive_onset = 0.5 * (1.0 - feats["mean_mood_14d"]) + 0.3 * (1.0 - feats["mean_resilience_14d"]) + 0.2 * (1.0 - feats["mean_social_14d"])
            depressive_onset = max(0.0, min(1.0, depressive_onset + np.random.normal(0, 0.04)))
            feats["depressive_label"] = depressive_onset
            
            all_data.append(feats)
            
    return pd.DataFrame(all_data)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # 1. Try to load Hugging Face survey baselines
    hf_baselines = try_load_huggingface_baselines()
    
    # 2. Simulate 1500 students * 7 days = 10,500 samples
    df = build_tabular_dataset(num_students=1500, hf_baselines=hf_baselines)
    
    # 3. Add random noise augmentation to features to improve robustness (\sigma = 0.05)
    feature_cols = [c for c in df.columns if c not in ["student_id", "day", "burnout_probability_target", "burnout_label", "anxiety_score_target", "anxiety_label", "depressive_label"]]
    for col in feature_cols:
        noise = np.random.normal(0, 0.02, len(df)) # Injected subtle noise (0.02 std deviation for clean scaling, maintaining target bounds)
        df[col] = np.clip(df[col] + noise, 0.0, 1.2) # Clip bounds for features
        
    print(f"Generated dataset shape: {df.shape}")
    print(f"Burnout class distribution:\n{df['burnout_label'].value_counts(normalize=True)}")
    print(f"Anxiety class distribution:\n{df['anxiety_label'].value_counts(normalize=True)}")
    print(f"Depressive onset average: {df['depressive_label'].mean():.4f}")
    
    # Save datasets
    df.to_csv("data/student_stress_dataset.csv", index=False)
    print("Saved student_stress_dataset.csv to data/")
    
    # Split into Train (80%) and Test (20%)
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print(f"Saved train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows) to data/")
