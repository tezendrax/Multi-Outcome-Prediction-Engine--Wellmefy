# MOPE Model Evaluation Report - Version v1.4.2

Generated on: 2026-07-12 14:43:25 UTC

This report summarizes the performance of the Multi-Outcome Prediction Engine (MOPE) ensemble models trained on the combined student cohort stress dataset.

## Target Performance Requirements
- **Burnout Classifier (XGBoost)**: AUC-ROC > 0.88, F1-Score > 0.85
- **Depressive Onset Regressor (LightGBM)**: MAE < 0.06

## Evaluation Results

### 1. Burnout Prediction (XGBoost)
- **Model Type**: XGBoost Binary Classifier
- **Training Duration**: Trained for **123 epochs** (boosting iterations / trees)
- **Parameters**: {'n_estimators': 123, 'max_depth': 7, 'learning_rate': 0.018443909450225893, 'subsample': 0.9789576469925393, 'colsample_bytree': 0.6747413885156027, 'scale_pos_weight': 1.0088842600587322, 'eval_metric': 'logloss', 'random_state': 42}
- **Test AUC-ROC**: **0.9800** (Target: > 0.88) - **PASSED**
- **Test F1-Score**: **0.9112** (Target: > 0.85) - **PASSED**

#### Classification Report:
```
              precision    recall  f1-score   support

  No Burnout       0.93      0.93      0.93      1150
     Burnout       0.91      0.91      0.91       950

    accuracy                           0.92      2100
   macro avg       0.92      0.92      0.92      2100
weighted avg       0.92      0.92      0.92      2100

```

#### Confusion Matrix:
![Burnout Confusion Matrix](confusion_matrix.png)


### 2. Anxiety Level Risk (XGBoost)
- **Model Type**: XGBoost Multiclass Classifier (0=Low, 1=Medium, 2=High)
- **Training Duration**: Trained for **150 epochs** (boosting iterations / trees)
- **Parameters**: {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.058966698206514886, 'subsample': 0.6902248971890871, 'random_state': 42}
- **Weighted F1-Score**: **0.7738**

#### Classification Report:
```
              precision    recall  f1-score   support

         Low       0.77      0.64      0.70       354
      Medium       0.75      0.81      0.78      1036
        High       0.82      0.79      0.80       710

    accuracy                           0.77      2100
   macro avg       0.78      0.75      0.76      2100
weighted avg       0.78      0.77      0.77      2100

```

### 3. Depressive Onset Index (LightGBM)
- **Model Type**: LightGBM Regressor (MAE L1 loss)
- **Training Duration**: Trained for **148 epochs** (boosting iterations / trees)
- **Parameters**: {'n_estimators': 148, 'max_depth': 3, 'num_leaves': 10, 'learning_rate': 0.06850639517928163, 'min_child_samples': 38, 'objective': 'regression_l1', 'random_state': 42, 'verbose': -1}
- **Test Mean Absolute Error (MAE)**: **0.0326** (Target: < 0.06) - **PASSED**

#### Regression Performance:
![Depressive Onset Actual vs Predicted](depressive_actual_vs_predicted.png)


## Feature Importance Summary (Top 10 features)

![Feature Importances](feature_importances.png)


| Feature Rank | Feature Name | Importance (Burnout XGBoost) |
|---|---|---|
| 1 | `latest_burnout` | 0.3031 |
| 2 | `mean_burnout_7d` | 0.2226 |
| 3 | `mean_burnout_14d` | 0.0901 |
| 4 | `std_burnout_14d` | 0.0389 |
| 5 | `std_burnout_7d` | 0.0219 |
| 6 | `mean_stress_14d` | 0.0160 |
| 7 | `mean_stress_7d` | 0.0127 |
| 8 | `mean_fatigue_14d` | 0.0105 |
| 9 | `mean_fatigue_7d` | 0.0104 |
| 10 | `mean_resilience_14d` | 0.0102 |
