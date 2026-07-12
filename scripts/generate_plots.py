import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_visualizations():
    print("Generating training diagnostic graphs...")
    
    # 1. Load data and models
    if not (os.path.exists("data/test.csv") and os.path.exists("data/models/burnout_xgb_v1.4.2.joblib")):
        print("Required test data or models not found. Make sure training has completed first.")
        return
        
    test_df = pd.read_csv("data/test.csv")
    model_version = "v1.4.2"
    
    burnout_model = joblib.load(f"data/models/burnout_xgb_{model_version}.joblib")
    depressive_model = joblib.load(f"data/models/depressive_lgb_{model_version}.joblib")
    features = joblib.load(f"data/models/features_list_{model_version}.joblib")
    
    X_test = test_df[features]
    y_test_burnout = test_df["burnout_label"]
    y_test_depressive = test_df["depressive_label"]
    
    # Set style
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'grid.color': '#333333',
        'text.color': '#FFFFFF',
        'axes.labelcolor': '#FFFFFF',
        'xtick.color': '#CCCCCC',
        'ytick.color': '#CCCCCC',
        'figure.facecolor': '#121212',
        'axes.facecolor': '#1E1E1E',
        'savefig.facecolor': '#121212',
        'font.size': 11
    })

    # ==========================================
    # Graph 1: Feature Importances (Burnout Model)
    # ==========================================
    plt.figure(figsize=(10, 6))
    importances = burnout_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    sns.barplot(
        x=importances[indices],
        y=[features[i] for i in indices],
        palette="viridis",
        hue=[features[i] for i in indices],
        legend=False
    )
    plt.title("Top 10 Feature Importances - Burnout Predictor", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Relative Importance Score", fontsize=12, labelpad=10)
    plt.ylabel("Engineered Feature Name", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig("data/feature_importances.png", dpi=300)
    plt.close()
    print("Saved data/feature_importances.png")

    # ==========================================
    # Graph 2: Confusion Matrix (Burnout Model)
    # ==========================================
    plt.figure(figsize=(8, 6))
    preds = burnout_model.predict(X_test)
    cm = confusion_matrix(y_test_burnout, preds)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["No Burnout", "Burnout"],
        yticklabels=["No Burnout", "Burnout"],
        annot_kws={"size": 14, "weight": "bold"},
        cbar=True
    )
    plt.title("Confusion Matrix - Burnout Classifier", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predicted Risk Class", fontsize=12, labelpad=10)
    plt.ylabel("True Risk Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig("data/confusion_matrix.png", dpi=300)
    plt.close()
    print("Saved data/confusion_matrix.png")

    # ==========================================
    # Graph 3: Actual vs Predicted (Depressive Onset Model)
    # ==========================================
    plt.figure(figsize=(8, 6))
    depressive_preds = depressive_model.predict(X_test)
    
    # Scatter plot
    sns.scatterplot(
        x=y_test_depressive, 
        y=depressive_preds, 
        alpha=0.6, 
        color="#00ADB5", 
        edgecolor=None
    )
    
    # Perfect alignment line
    plt.plot([0, 1], [0, 1], color="#FF2E93", linestyle="--", linewidth=2, label="Perfect Prediction")
    
    plt.title("Actual vs. Predicted - Depressive Onset Index", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Ground Truth Index [0.0 - 1.0]", fontsize=12, labelpad=10)
    plt.ylabel("Model Predicted Index [0.0 - 1.0]", fontsize=12, labelpad=10)
    plt.xlim(0.1, 0.8)
    plt.ylim(0.1, 0.8)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("data/depressive_actual_vs_predicted.png", dpi=300)
    plt.close()
    print("Saved data/depressive_actual_vs_predicted.png")
    print("All diagnostic graphs generated successfully.")

if __name__ == "__main__":
    generate_visualizations()
