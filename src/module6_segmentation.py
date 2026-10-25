"""
Module 6: Member Behavioral Segmentation (Phase 6)
Aggregates interactions at the Member Level, performs K-Means clustering,
validates with Silhouette and Davies-Bouldin scores, conducts PCA (2 components),
profiles behavioral personas, and appends Member_Segment to member & interaction data.
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
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


def aggregate_member_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates interaction metrics to the unique Member level.
    """
    member_agg = df.groupby("Member_ID").agg(
        Age=("Age", "first"),
        Plan_Type=("Plan_Type", "first"),
        Region=("Region", "first"),
        Contact_Frequency_Score=("Contact_Frequency_Score", "first"),
        Digital_Adoption_Score=("Digital_Adoption_Score", "first"),
        Repeat_Contact_Rate=("Repeat_Contact_Rate", "first"),
        Avg_Service_Friction=("Service_Friction_Index", "mean"),
        Avg_Resolution_Complexity=("Resolution_Complexity_Index", "mean"),
        Member_Escalation_Rate=("Escalation_Flag", "mean"),
        Member_Mean_CSAT=("Satisfaction_Score", "mean"),
        Total_Lifetime_Interactions=("Interaction_ID", "count")
    ).reset_index()

    print(f"  [AGGREGATION] Aggregated {len(df)} interactions into {len(member_agg)} unique member profiles.")
    return member_agg


def evaluate_optimal_clusters(X_scaled: np.ndarray, k_range=range(3, 7)) -> dict:
    """
    Evaluates cluster compactness and separation across candidate k values.
    """
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        scores[k] = {
            "silhouette_score": round(float(sil), 4),
            "davies_bouldin_index": round(float(db), 4),
            "calinski_harabasz_score": round(float(ch), 2)
        }
        print(f"    * k={k}: Silhouette = {sil:.4f}, Davies-Bouldin = {db:.4f}, Calinski-Harabasz = {ch:.1f}")

    return scores


def assign_persona_labels(cluster_centers_df: pd.DataFrame) -> dict:
    """
    Assigns intuitive Kaiser healthcare personas based on cluster center attributes.
    """
    persona_map = {}
    for idx, row in cluster_centers_df.iterrows():
        dig = row["Digital_Adoption_Score"]
        fric = row["Avg_Service_Friction"]
        esc = row["Member_Escalation_Rate"]
        freq = row["Contact_Frequency_Score"]

        if dig >= 0.60 and fric < 0.35:
            persona = "Digital-First Self-Service"
        elif esc >= 0.25 or fric >= 0.40:
            persona = "Escalation-Prone / High-Friction"
        elif freq >= 0.25 or row["Age"] >= 58:
            persona = "Service-Intensive / High-Touch"
        else:
            persona = "Routine Low-Need Care Seekers"

        # Handle duplicates if rules collide
        existing = list(persona_map.values())
        if persona in existing:
            persona = f"{persona} (Sub-group {idx})"
        persona_map[idx] = persona

    return persona_map


def perform_segmentation(df_member: pd.DataFrame, n_clusters: int = 4):
    """
    Executes StandardScaler, K-Means clustering, PCA visualization, and profiling.
    """
    features = [
        "Age",
        "Contact_Frequency_Score",
        "Digital_Adoption_Score",
        "Repeat_Contact_Rate",
        "Avg_Service_Friction",
        "Avg_Resolution_Complexity",
        "Member_Escalation_Rate"
    ]

    X = df_member[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n  [EVALUATION] Testing cluster solutions k=3 to k=6:")
    cluster_scores = evaluate_optimal_clusters(X_scaled)

    # Train optimal KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)
    df_member["Cluster_ID"] = cluster_labels

    # Compute unscaled cluster centers for business persona interpretation
    centers_df = df_member.groupby("Cluster_ID")[features].mean()
    persona_mapping = assign_persona_labels(centers_df)
    df_member["Member_Segment"] = df_member["Cluster_ID"].map(persona_mapping)

    print("\n  [CLUSTER PROFILES & PERSONAS]")
    for cid, persona in persona_mapping.items():
        c_stats = centers_df.loc[cid]
        c_size = (df_member["Cluster_ID"] == cid).sum()
        c_pct = c_size / len(df_member) * 100
        print(f"  Cluster {cid} -> '{persona}': {c_size} members ({c_pct:.1f}%)")
        print(f"      Age: {c_stats['Age']:.1f} | Digital: {c_stats['Digital_Adoption_Score']*100:.1f}% | Friction: {c_stats['Avg_Service_Friction']:.3f} | Esc Rate: {c_stats['Member_Escalation_Rate']*100:.1f}%")

    # PCA 2-Component Visualization
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    df_member["PCA1"] = X_pca[:, 0]
    df_member["PCA2"] = X_pca[:, 1]
    var_exp = pca.explained_variance_ratio_

    plot_cluster_pca(df_member, persona_mapping, var_exp)

    # Save Model Artifacts
    segmentation_artifacts = {
        "kmeans": kmeans,
        "scaler": scaler,
        "features": features,
        "persona_mapping": persona_mapping,
        "pca": pca
    }
    model_path = MODELS_DIR / "kmeans_segmentation.pkl"
    joblib.dump(segmentation_artifacts, model_path)
    print(f"  [MODEL EXPORT] Saved KMeans segmentation artifact to: {model_path}")

    # Export Segmentation Report
    seg_report = {
        "n_clusters": n_clusters,
        "cluster_evaluations": cluster_scores,
        "cluster_centers": centers_df.to_dict(orient="index"),
        "persona_mapping": {str(k): v for k, v in persona_mapping.items()},
        "pca_explained_variance_ratio": [round(float(v), 4) for v in var_exp]
    }
    report_path = REPORTS_DIR / "segmentation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(seg_report, f, indent=4)
    print(f"  [REPORT SAVED] Exported segmentation report to: {report_path}")

    return df_member, segmentation_artifacts


def plot_cluster_pca(df_member: pd.DataFrame, persona_mapping: dict, var_exp: np.ndarray):
    """Plots 2D PCA representation of member behavioral clusters."""
    plt.figure(figsize=(11, 8))
    
    palette = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6", "#f39c12"]
    unique_clusters = sorted(df_member["Cluster_ID"].unique())

    for i, cid in enumerate(unique_clusters):
        sub = df_member[df_member["Cluster_ID"] == cid]
        persona = persona_mapping[cid]
        color = palette[i % len(palette)]
        plt.scatter(
            sub["PCA1"], sub["PCA2"],
            label=f"Cluster {cid}: {persona} (n={len(sub):,})",
            alpha=0.65, s=40, color=color, edgecolors="none"
        )

        # Plot cluster center in PCA space
        center_x = sub["PCA1"].mean()
        center_y = sub["PCA2"].mean()
        plt.scatter(center_x, center_y, color=color, s=180, edgecolor="black", linewidth=2, marker="X")

    plt.title(
        f"Kaiser Permanente Member Behavioral Segmentation (PCA 2D Projection)\n"
        f"PC1: {var_exp[0]*100:.1f}% Variance | PC2: {var_exp[1]*100:.1f}% Variance",
        fontsize=13, fontweight="bold", pad=14
    )
    plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% var)", fontsize=11)
    plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% var)", fontsize=11)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()

    output_path = FIGURES_DIR / "segmentation_clusters_pca.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")


def run_module6(df_interactions: pd.DataFrame):
    """Execution entry point for Module 6."""
    print("\n>>> EXECUTING MODULE 6: MEMBER BEHAVIORAL SEGMENTATION <<<")
    df_member = aggregate_member_profiles(df_interactions)
    df_member_enriched, artifacts = perform_segmentation(df_member, n_clusters=4)

    # Join Member_Segment back to interaction level dataset
    df_interactions_enriched = df_interactions.merge(
        df_member_enriched[["Member_ID", "Member_Segment"]],
        on="Member_ID",
        how="left"
    )
    print(f"  [JOIN] Appended Member_Segment to all {len(df_interactions_enriched)} interaction records.")

    return df_interactions_enriched, df_member_enriched, artifacts


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    from src.module4_feature_engineering import run_module4
    from src.module5_escalation_ml import run_module5

    df_data, _ = run_module1()
    df_feat = run_module4(df_data)
    df_scored, _, _ = run_module5(df_feat)
    df_final, df_mem, _ = run_module6(df_scored)
