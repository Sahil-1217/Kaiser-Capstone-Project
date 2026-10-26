"""
Module 5: Escalation Prediction Machine Learning (Phase 5)
Trains, cross-validates, and benchmarks leakage-free ML models for Escalation Prediction:
- Strict Data Leakage Prevention (Pre-interaction features only)
- Models: Logistic Regression, Random Forest, HistGradientBoosting
- Stratified 80/20 Split & 5-Fold Stratified Cross Validation
- Comprehensive Evaluation (Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score)
- Probability generation (Predicted_Escalation_Prob) and artifact export
"""

import os
import sys
import json
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve, brier_score_loss, confusion_matrix,
    accuracy_score
)
from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


# Strict Feature Separation
PRE_INTERACTION_NUMERIC = [
    "Age",
    "Total_Contacts",
    "Contact_Frequency_Score",
    "Digital_Adoption_Score",
    "Repeat_Contact_Rate",
    "Member_Interaction_Count"
]

PRE_INTERACTION_CATEGORICAL = [
    "Plan_Type",
    "Region",
    "Contact_Channel",
    "Interaction_Reason",
    "Queue",
    "Department",
    "Shift"
]

# Explicit Leakage Exclusion List
EXCLUDED_POST_INTERACTION_COLUMNS = [
    "Resolution_Time",
    "First_Contact_Resolution",
    "Satisfaction_Score",
    "Service_Friction_Index",
    "Resolution_Complexity_Index",
    "Escalation_Flag",
    "Interaction_ID",
    "Member_ID",
    "Agent_ID",
    "Interaction_Date",
    "Call_Transcript",
    "Chat_Transcript",
    "Feedback_Comments"
]


def prepare_features(df: pd.DataFrame):
    """
    Extracts strictly pre-interaction features and target, verifying zero leakage.
    """
    y = df["Escalation_Flag"].astype(int)
    
    # Verify no post-interaction variables exist in feature set
    X = df[PRE_INTERACTION_NUMERIC + PRE_INTERACTION_CATEGORICAL].copy()
    for col in EXCLUDED_POST_INTERACTION_COLUMNS:
        assert col not in X.columns or col == "Escalation_Flag", f"LEAKAGE ALERT: {col} found in feature matrix!"

    print(f"  [DATA LEAKAGE SAFEGUARD] Verified {len(PRE_INTERACTION_NUMERIC)} numeric and {len(PRE_INTERACTION_CATEGORICAL)} categorical pre-interaction features.")
    print("  [LEAKAGE EXCLUSIONS CONFIRMED] Excluded Resolution_Time, FCR, Satisfaction_Score, and Friction Indices.")
    return X, y


def build_preprocessor():
    """Builds scikit-learn ColumnTransformer for pre-interaction features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), PRE_INTERACTION_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PRE_INTERACTION_CATEGORICAL)
        ]
    )


def train_and_evaluate_models(X: pd.DataFrame, y: pd.Series):
    """
    Performs 80/20 Stratified train/test split, 5-fold cross validation,
    and comprehensive evaluation across 3 algorithms.
    """
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  [SPLIT] Train set: {len(X_train)} samples ({y_train.mean()*100:.1f}% positive), Test set: {len(X_test)} samples ({y_test.mean()*100:.1f}% positive)")

    preprocessor = build_preprocessor()

    # Model definitions
    models = {
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision_Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient_Boosting": HistGradientBoostingClassifier(
            max_iter=120,
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    model_results = {}
    fitted_pipelines = {}

    for name, clf in models.items():
        print(f"\n  [TRAINING] Benchmarking {name} with 5-Fold Cross Validation...")
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # 5-Fold CV on Train set
        scoring = ["roc_auc", "average_precision", "f1", "precision", "recall", "accuracy"]
        cv_scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        # Fit on full training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Test Set Evaluation
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        brier = brier_score_loss(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        model_results[name] = {
            "cv_mean_accuracy": round(float(np.mean(cv_scores["test_accuracy"])), 4),
            "cv_mean_roc_auc": round(float(np.mean(cv_scores["test_roc_auc"])), 4),
            "cv_std_roc_auc": round(float(np.std(cv_scores["test_roc_auc"])), 4),
            "cv_mean_pr_auc": round(float(np.mean(cv_scores["test_average_precision"])), 4),
            "cv_mean_f1": round(float(np.mean(cv_scores["test_f1"])), 4),
            "test_accuracy": round(float(acc), 4),
            "test_roc_auc": round(float(roc_auc), 4),
            "test_pr_auc": round(float(pr_auc), 4),
            "test_brier_score": round(float(brier), 4),
            "test_precision_pos": round(float(report["1"]["precision"]), 4),
            "test_recall_pos": round(float(report["1"]["recall"]), 4),
            "test_f1_pos": round(float(report["1"]["f1-score"]), 4),
            "confusion_matrix": cm,
            "y_prob_test": y_prob
        }

        print(f"    * 5-Fold CV Accuracy: {model_results[name]['cv_mean_accuracy']} | ROC-AUC: {model_results[name]['cv_mean_roc_auc']} (+/- {model_results[name]['cv_std_roc_auc']})")
        print(f"    * Test Accuracy:     {model_results[name]['test_accuracy']}")
        print(f"    * Test ROC-AUC:      {model_results[name]['test_roc_auc']}")
        print(f"    * Test PR-AUC:       {model_results[name]['test_pr_auc']}")
        print(f"    * Test Recall (Esc): {model_results[name]['test_recall_pos']}")
        print(f"    * Test F1-Score:     {model_results[name]['test_f1_pos']}")

    # Identify Best Model by Test PR-AUC and ROC-AUC
    best_model_name = max(model_results, key=lambda k: model_results[k]["test_roc_auc"] + model_results[k]["test_pr_auc"])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"\n  [CHAMPION MODEL] Selected '{best_model_name}' as optimal escalation predictor.")

    # Save Best Model Artifact
    model_artifact_path = MODELS_DIR / "escalation_model.pkl"
    joblib.dump(best_pipeline, model_artifact_path)
    print(f"  [MODEL EXPORT] Saved champion model artifact to: {model_artifact_path}")

    # Plot ROC & PR Curves
    plot_evaluation_curves(model_results, y_test)

    # Clean results for JSON export (remove numpy arrays)
    exportable_results = {}
    for k, v in model_results.items():
        exportable_results[k] = {ik: iv for ik, iv in v.items() if ik != "y_prob_test"}
    exportable_results["champion_model"] = best_model_name

    json_path = REPORTS_DIR / "ml_model_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exportable_results, f, indent=4)
    print(f"  [REPORT SAVED] Exported model metrics to: {json_path}")

    return best_pipeline, exportable_results


def plot_evaluation_curves(model_results: dict, y_test: pd.Series):
    """Generates comparative ROC and Precision-Recall Curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        "Logistic_Regression": "#2ca02c",
        "Decision_Tree": "#9467bd",
        "Random_Forest": "#1f77b4",
        "Gradient_Boosting": "#ff7f0e"
    }

    # 1. ROC Curves
    for name, res in model_results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob_test"])
        axes[0].plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC = {res['test_roc_auc']:.3f})", color=colors.get(name, "black"), linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (0.500)")
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)

    # 2. Precision-Recall Curves
    baseline_pr = y_test.mean()
    for name, res in model_results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob_test"])
        axes[1].plot(rec, prec, label=f"{name.replace('_', ' ')} (PR-AUC = {res['test_pr_auc']:.3f})", color=colors.get(name, "black"), linewidth=2)

    axes[1].axhline(baseline_pr, color="k", linestyle="--", alpha=0.5, label=f"No-Skill Baseline ({baseline_pr:.3f})")
    axes[1].set_title("Precision-Recall (PR) Curve (Class-Imbalance Aware)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("Recall (Coverage)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    output_path = FIGURES_DIR / "ml_roc_pr_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")


def append_predicted_probabilities(pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores the full interaction dataset with the trained model and appends
    Predicted_Escalation_Prob.
    """
    df = df.copy()
    X = df[PRE_INTERACTION_NUMERIC + PRE_INTERACTION_CATEGORICAL]
    probs = pipeline.predict_proba(X)[:, 1]
    df["Predicted_Escalation_Prob"] = np.round(probs, 4)
    print(f"  [INFERENCE] Appended Predicted_Escalation_Prob to dataset (Mean: {df['Predicted_Escalation_Prob'].mean():.4f}, Max: {df['Predicted_Escalation_Prob'].max():.4f})")
    return df


def run_module5(df: pd.DataFrame) -> tuple:
    """Execution entry point for Module 5."""
    print("\n>>> EXECUTING MODULE 5: ESCALATION PREDICTION MACHINE LEARNING <<<")
    X, y = prepare_features(df)
    champion_pipeline, eval_metrics = train_and_evaluate_models(X, y)
    df_scored = append_predicted_probabilities(champion_pipeline, df)
    return df_scored, champion_pipeline, eval_metrics


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    from src.module4_feature_engineering import run_module4
    df_data, _ = run_module1()
    df_feat = run_module4(df_data)
    df_scored, _, _ = run_module5(df_feat)
