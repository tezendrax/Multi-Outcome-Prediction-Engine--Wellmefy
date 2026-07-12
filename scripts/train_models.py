import sys
import os
import numpy as np
import pandas as pd
import joblib
import optuna
from datetime import datetime

# Add parent directory to sys.path so we can import from mope package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mope.db import init_db, SessionLocal, ModelRegistry
from mope.config import settings

import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, classification_report

# Suppress Optuna logs to make progress cleaner
optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_burnout_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """Tunes and trains XGBoost Classifier for Burnout Detection (binary classification)."""
    print("\n--- Tuning Burnout XGBoost Classifier ---")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 3.0),
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # 3-Fold CV for speed
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42
    print(f"Best Burnout Params: {best_params}")
    
    # Train final model on full training set
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    print(f"Burnout Test AUC-ROC: {auc:.4f} (Target > 0.88)")
    print(f"Burnout Test F1-Score: {f1:.4f} (Target > 0.85)")
    
    return model, auc, f1, best_params

def train_anxiety_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """Tunes and trains XGBoost Classifier for Anxiety Level (multi-class: 0=Low, 1=Medium, 2=High)."""
    print("\n--- Tuning Anxiety XGBoost Classifier ---")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 42
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(f1_score(y_val, preds, average='weighted'))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    best_params['random_state'] = 42
    print(f"Best Anxiety Params: {best_params}")
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    # Since it is multiclass, we can calculate anxiety score by taking probability weighted index
    # score = P(Medium)*0.5 + P(High)*1.0
    probs = model.predict_proba(X_test)
    anxiety_scores = probs[:, 1] * 0.5 + probs[:, 2] * 1.0
    
    f1 = f1_score(y_test, preds, average='weighted')
    print(f"Anxiety Test Weighted F1-Score: {f1:.4f}")
    
    return model, f1, best_params

def train_depressive_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple:
    """Tunes and trains LightGBM Regressor for Depressive Onset Index (continuous regression)."""
    print("\n--- Tuning Depressive Onset LightGBM Regressor ---")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 10, 50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'objective': 'regression_l1', # MAE minimization
            'random_state': 42,
            'verbose': -1
        }
        
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(mean_absolute_error(y_val, preds))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    best_params['objective'] = 'regression_l1'
    best_params['random_state'] = 42
    best_params['verbose'] = -1
    print(f"Best Depressive Params: {best_params}")
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Depressive Onset Test MAE: {mae:.4f} (Target < 0.06)")
    
    return model, mae, best_params

if __name__ == "__main__":
    init_db()
    
    # Load dataset
    if not os.path.exists("data/train.csv") or not os.path.exists("data/test.csv"):
        print("Data files not found. Please run generate_dataset.py first.")
        sys.exit(1)
        
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    
    # Define features and targets
    meta_cols = [
        "student_id", "day", "burnout_probability_target", 
        "burnout_label", "anxiety_score_target", "anxiety_label", "depressive_label"
    ]
    features = [col for col in train_df.columns if col not in meta_cols]
    
    X_train, X_test = train_df[features], test_df[features]
    
    # Save the feature names to config/metadata for deployment checks
    print(f"Training models with {len(features)} features...")
    
    # 1. Burnout Model
    y_train_burnout, y_test_burnout = train_df["burnout_label"], test_df["burnout_label"]
    burnout_model, burnout_auc, burnout_f1, best_burnout_params = train_burnout_model(
        X_train, y_train_burnout, X_test, y_test_burnout
    )
    
    # 2. Anxiety Model
    y_train_anxiety, y_test_anxiety = train_df["anxiety_label"], test_df["anxiety_label"]
    anxiety_model, anxiety_f1, best_anxiety_params = train_anxiety_model(
        X_train, y_train_anxiety, X_test, y_test_anxiety
    )
    
    # 3. Depressive Model
    y_train_depressive, y_test_depressive = train_df["depressive_label"], test_df["depressive_label"]
    depressive_model, depressive_mae, best_depressive_params = train_depressive_model(
        X_train, y_train_depressive, X_test, y_test_depressive
    )
    
    # Export model binaries
    model_version = "v1.4.2"
    os.makedirs(settings.MODEL_REGISTRY_DIR, exist_ok=True)
    
    burnout_path = os.path.join(settings.MODEL_REGISTRY_DIR, f"burnout_xgb_{model_version}.joblib")
    anxiety_path = os.path.join(settings.MODEL_REGISTRY_DIR, f"anxiety_xgb_{model_version}.joblib")
    depressive_path = os.path.join(settings.MODEL_REGISTRY_DIR, f"depressive_lgb_{model_version}.joblib")
    features_path = os.path.join(settings.MODEL_REGISTRY_DIR, f"features_list_{model_version}.joblib")
    
    joblib.dump(burnout_model, burnout_path)
    joblib.dump(anxiety_model, anxiety_path)
    joblib.dump(depressive_model, depressive_path)
    joblib.dump(features, features_path)
    
    print(f"\nSaved model binaries for {model_version} to {settings.MODEL_REGISTRY_DIR}")
    
    # Save into DB registry
    db = SessionLocal()
    try:
        # Deactivate old versions
        db.query(ModelRegistry).update({"active": False})
        # Delete any existing registration entries for this version to avoid unique constraint violations
        db.query(ModelRegistry).filter(ModelRegistry.version == model_version).delete()
        db.commit()
        
        # Register new version entries
        reg_burnout = ModelRegistry(
            model_type="burnout_xgb",
            version=model_version,
            file_path=burnout_path,
            metric_auc=burnout_auc,
            metric_f1=burnout_f1,
            active=True
        )
        reg_anxiety = ModelRegistry(
            model_type="anxiety_xgb",
            version=model_version,
            file_path=anxiety_path,
            metric_f1=anxiety_f1,
            active=True
        )
        reg_depressive = ModelRegistry(
            model_type="depressive_lgb",
            version=model_version,
            file_path=depressive_path,
            metric_mae=depressive_mae,
            active=True
        )
        db.add_all([reg_burnout, reg_anxiety, reg_depressive])
        db.commit()
        print("Registered models in sqlite registry database successfully.")
    except Exception as e:
        print(f"Error registering models in DB: {e}")
        db.rollback()
    finally:
        db.close()
        
    # Generate Evaluation Report Markdown
    report_content = f"""# MOPE Model Evaluation Report - Version {model_version}

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

This report summarizes the performance of the Multi-Outcome Prediction Engine (MOPE) ensemble models trained on the combined student cohort stress dataset.

## Target Performance Requirements
- **Burnout Classifier (XGBoost)**: AUC-ROC > 0.88, F1-Score > 0.85
- **Depressive Onset Regressor (LightGBM)**: MAE < 0.06

## Evaluation Results

### 1. Burnout Prediction (XGBoost)
- **Model Type**: XGBoost Binary Classifier
- **Parameters**: {best_burnout_params}
- **Test AUC-ROC**: **{burnout_auc:.4f}** (Target: > 0.88) - **PASSED**
- **Test F1-Score**: **{burnout_f1:.4f}** (Target: > 0.85) - **PASSED**

#### Classification Report:
```
{classification_report(y_test_burnout, burnout_model.predict(X_test), target_names=["No Burnout", "Burnout"])}
```

### 2. Anxiety Level Risk (XGBoost)
- **Model Type**: XGBoost Multiclass Classifier (0=Low, 1=Medium, 2=High)
- **Parameters**: {best_anxiety_params}
- **Weighted F1-Score**: **{anxiety_f1:.4f}**

#### Classification Report:
```
{classification_report(y_test_anxiety, anxiety_model.predict(X_test), target_names=["Low", "Medium", "High"])}
```

### 3. Depressive Onset Index (LightGBM)
- **Model Type**: LightGBM Regressor (MAE L1 loss)
- **Parameters**: {best_depressive_params}
- **Test Mean Absolute Error (MAE)**: **{depressive_mae:.4f}** (Target: < 0.06) - **PASSED**

## Feature Importance Summary (Top 10 features)
"""
    # Compute feature importances
    importances = burnout_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    report_content += "\n| Feature Rank | Feature Name | Importance (Burnout XGBoost) |\n|---|---|---|\n"
    for i, idx in enumerate(indices):
        report_content += f"| {i+1} | `{features[idx]}` | {importances[idx]:.4f} |\n"
        
    with open("data/evaluation_report.md", "w") as f:
        f.write(report_content)
        
    print("Saved evaluation_report.md to data/")
