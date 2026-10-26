"""
Module 3: Exploratory Data Analysis (EDA) Automation (Phase 3)
Automates distribution checks, summary statistics, and saves publication-grade
analytical figures and tabular breakdowns:
1. eda_distributions.png (Resolution_Time, Satisfaction_Score, Total_Contacts)
2. eda_correlations.png (Correlation matrix heatmap)
3. eda_channel_escalation.png (Channel vs Escalation Rate & FCR)
4. eda_reason_resolution.png (Interaction Reason vs Resolution Time & CSAT)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import FIGURES_DIR, REPORTS_DIR

# Styling setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def generate_distribution_plots(df: pd.DataFrame):
    """Generate multi-panel distribution plots for core operational variables."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Resolution Time
    sns.histplot(df["Resolution_Time"], kde=True, ax=axes[0], color="#1f77b4", bins=30, edgecolor="white")
    axes[0].set_title("Resolution Time Distribution (min)", fontsize=13, fontweight="bold", pad=12)
    axes[0].set_xlabel("Duration (Minutes)", fontsize=11)
    axes[0].set_ylabel("Interaction Count", fontsize=11)
    med_rt = df["Resolution_Time"].median()
    axes[0].axvline(med_rt, color="#d62728", linestyle="--", linewidth=1.5, label=f"Median: {med_rt:.1f}m")
    axes[0].legend(frameon=True)

    # 2. Satisfaction Score
    csat_counts = df["Satisfaction_Score"].value_counts().sort_index()
    palette = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]
    axes[1].bar(csat_counts.index, csat_counts.values, color=palette, edgecolor="black", alpha=0.85, width=0.6)
    axes[1].set_title("Member Satisfaction Score (CSAT 1-5)", fontsize=13, fontweight="bold", pad=12)
    axes[1].set_xlabel("CSAT Rating", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    for x, val in zip(csat_counts.index, csat_counts.values):
        axes[1].text(x, val + 50, f"{val:,}", ha="center", fontsize=9, fontweight="bold")

    # 3. Total Contacts (Contact Frequency)
    contact_counts = df["Total_Contacts"].value_counts().sort_index()
    axes[2].bar(contact_counts.index, contact_counts.values, color="#9467bd", edgecolor="black", alpha=0.85, width=0.6)
    axes[2].set_title("Total Contacts per Interaction Inquiry", fontsize=13, fontweight="bold", pad=12)
    axes[2].set_xlabel("Number of Contacts", fontsize=11)
    axes[2].set_ylabel("Frequency", fontsize=11)
    for x, val in zip(contact_counts.index, contact_counts.values):
        axes[2].text(x, val + 60, f"{val:,}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path = FIGURES_DIR / "eda_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")


def generate_correlation_matrix(df: pd.DataFrame):
    """Generate and save correlation matrix heatmap of numerical metrics."""
    numeric_cols = [
        "Age", "Total_Contacts", "Resolution_Time",
        "Escalation_Flag", "Satisfaction_Score", "First_Contact_Resolution"
    ]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.8, linecolor="white", cbar_kws={"shrink": 0.8}
    )
    plt.title("Operational Metrics Correlation Heatmap", fontsize=14, fontweight="bold", pad=16)
    plt.tight_layout()

    output_path = FIGURES_DIR / "eda_correlations.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")


def generate_channel_breakdown_plot(df: pd.DataFrame):
    """Generate bivariate breakdown: Channel vs. Escalation Rate & FCR."""
    grouped = df.groupby("Contact_Channel").agg(
        Total_Volume=("Interaction_ID", "count"),
        Escalation_Rate=("Escalation_Flag", "mean"),
        FCR_Rate=("First_Contact_Resolution", "mean"),
        Mean_CSAT=("Satisfaction_Score", "mean"),
        Mean_Resolution_Time=("Resolution_Time", "mean")
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(grouped))
    width = 0.35

    bars1 = ax1.bar(x - width/2, grouped["Escalation_Rate"] * 100, width, label="Escalation Rate (%)", color="#e74c3c", edgecolor="black")
    bars2 = ax1.bar(x + width/2, grouped["FCR_Rate"] * 100, width, label="First Contact Resolution (%)", color="#2ecc71", edgecolor="black")

    ax1.set_xlabel("Contact Channel", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Channel Performance: Escalation Rate vs. First Contact Resolution (FCR)", fontsize=13, fontweight="bold", pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(grouped["Contact_Channel"], fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", frameon=True)

    # Annotate bar values
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")
    for b in bars2:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    output_path = FIGURES_DIR / "eda_channel_escalation.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")

    # Save summary table
    table_path = REPORTS_DIR / "eda_channel_summary.csv"
    grouped.to_csv(table_path, index=False)
    print(f"  [TABLE SAVED] {table_path.name}")


def generate_reason_resolution_plot(df: pd.DataFrame):
    """Generate breakdown: Interaction Reason vs Resolution Time and CSAT."""
    reason_stats = df.groupby("Interaction_Reason").agg(
        Mean_Resolution_Time=("Resolution_Time", "mean"),
        Median_Resolution_Time=("Resolution_Time", "median"),
        Escalation_Rate=("Escalation_Flag", "mean"),
        Mean_CSAT=("Satisfaction_Score", "mean"),
        Volume=("Interaction_ID", "count")
    ).sort_values("Mean_Resolution_Time", ascending=True).reset_index()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    y = np.arange(len(reason_stats))
    ax1.barh(y, reason_stats["Mean_Resolution_Time"], color="#3498db", edgecolor="black", alpha=0.85, height=0.55)
    ax1.set_yticks(y)
    ax1.set_yticklabels(reason_stats["Interaction_Reason"], fontsize=10)
    ax1.set_xlabel("Mean Resolution Time (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Resolution Time by Interaction Reason", fontsize=13, fontweight="bold", pad=14)

    for i, val in enumerate(reason_stats["Mean_Resolution_Time"]):
        ax1.text(val + 0.3, i, f"{val:.1f}m (CSAT: {reason_stats.loc[i, 'Mean_CSAT']:.2f})", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    output_path = FIGURES_DIR / "eda_reason_resolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")

    table_path = REPORTS_DIR / "eda_reason_summary.csv"
    reason_stats.to_csv(table_path, index=False)
    print(f"  [TABLE SAVED] {table_path.name}")


def generate_time_trends_plot(df: pd.DataFrame):
    """
    Generate chronological time trends: Monthly interaction volume and escalation rate.
    Provides operational visibility into seasonal peaks and grievance surges.
    """
    df_time = df.copy()
    if "Interaction_Date" in df_time.columns:
        df_time["Date"] = pd.to_datetime(df_time["Interaction_Date"], errors="coerce")
    else:
        # Fallback date generation if not directly present
        df_time["Date"] = pd.date_range(start="2024-01-01", periods=len(df_time), freq="h")

    df_time["YearMonth"] = df_time["Date"].dt.to_period("M").astype(str)
    
    monthly = df_time.groupby("YearMonth").agg(
        Total_Volume=("Interaction_ID", "count"),
        Escalation_Rate=("Escalation_Flag", lambda x: np.mean(x) * 100.0),
        FCR_Rate=("First_Contact_Resolution", lambda x: np.mean(x) * 100.0),
        Mean_CSAT=("Satisfaction_Score", "mean"),
        Mean_Duration=("Resolution_Time", "mean")
    ).reset_index().sort_values("YearMonth")

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(monthly))
    bars = ax1.bar(x, monthly["Total_Volume"], color="#005A9C", alpha=0.75, width=0.55, label="Monthly Volume", edgecolor="black")
    ax1.set_xlabel("Operational Period (Year-Month)", fontsize=11, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Interaction Volume", fontsize=11, fontweight="bold", color="#003B71")
    ax1.set_xticks(x)
    ax1.set_xticklabels(monthly["YearMonth"], rotation=35, ha="right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    line = ax2.plot(x, monthly["Escalation_Rate"], color="#B91C1C", linewidth=2.5, marker="o", markersize=6, label="Escalation Rate (%)")
    ax2.set_ylabel("Escalation Rate (%)", fontsize=11, fontweight="bold", color="#B91C1C")
    ax2.set_ylim(0, max(monthly["Escalation_Rate"].max() * 1.3, 35))
    ax2.grid(False)

    plt.title("Kaiser Permanente Longitudinal Operational Trends: Volume vs. Escalation Rate", fontsize=13, fontweight="bold", pad=14)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    output_path = FIGURES_DIR / "eda_time_trends.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")

    summary_path = REPORTS_DIR / "eda_time_trends_summary.csv"
    monthly.to_csv(summary_path, index=False)
    print(f"  [TABLE SAVED] {summary_path.name}")


def generate_outlier_plots(df: pd.DataFrame):
    """
    Generate multi-panel boxplots to identify operational outliers across
    duration, total contacts, and age. Exports IQR statistical thresholds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    target_cols = [
        ("Resolution_Time", "Resolution Duration (min)", "#1f77b4"),
        ("Total_Contacts", "Total Contacts (Count)", "#2ca02c"),
        ("Age", "Member Age (Years)", "#ff7f0e")
    ]

    outlier_summary = []

    for idx, (col, label, color) in enumerate(target_cols):
        if col not in df.columns:
            continue
        
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_pct = len(outliers) / len(series) * 100.0

        outlier_summary.append({
            "Metric": col,
            "Q1_25Pct": round(float(q1), 2),
            "Median": round(float(series.median()), 2),
            "Q3_75Pct": round(float(q3), 2),
            "IQR": round(float(iqr), 2),
            "Lower_Fence": round(float(lower_bound), 2),
            "Upper_Fence": round(float(upper_bound), 2),
            "Outlier_Count": int(len(outliers)),
            "Outlier_Pct": round(float(outlier_pct), 2)
        })

        sns.boxplot(
            y=series, ax=axes[idx], color=color,
            flierprops={"marker": "d", "markersize": 4, "markerfacecolor": "#d62728", "alpha": 0.6}
        )
        axes[idx].set_title(f"{label}\nOutliers: {len(outliers):,} ({outlier_pct:.1f}%)", fontsize=11, fontweight="bold")
        axes[idx].set_ylabel(label, fontsize=10)
        axes[idx].axhline(upper_bound, color="#d62728", linestyle="--", linewidth=1.2, label=f"Upper Fence: {upper_bound:.1f}")
        axes[idx].legend(loc="upper right", fontsize=8)

    plt.suptitle("Kaiser Permanente Operational Outlier Analysis (1.5 × IQR Fences)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = FIGURES_DIR / "eda_outliers_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")

    outlier_df = pd.DataFrame(outlier_summary)
    summary_path = REPORTS_DIR / "eda_outliers_summary.csv"
    outlier_df.to_csv(summary_path, index=False)
    print(f"  [TABLE SAVED] {summary_path.name}")


def generate_summary_tables(df: pd.DataFrame) -> dict:
    """Compute and export comprehensive numerical descriptive statistics."""
    summary = df[["Resolution_Time", "Satisfaction_Score", "Total_Contacts", "Age"]].describe().T
    summary["IQR"] = summary["75%"] - summary["25%"]
    summary["Skewness"] = df[["Resolution_Time", "Satisfaction_Score", "Total_Contacts", "Age"]].skew()
    summary["Kurtosis"] = df[["Resolution_Time", "Satisfaction_Score", "Total_Contacts", "Age"]].kurtosis()

    summary_path = REPORTS_DIR / "eda_descriptive_statistics.csv"
    summary.to_csv(summary_path)
    print(f"  [TABLE SAVED] {summary_path.name}")

    return summary.to_dict()


def run_module3(df: pd.DataFrame):
    """Execution entry point for Module 3."""
    print("\n>>> EXECUTING MODULE 3: EXPLORATORY DATA ANALYSIS (EDA) AUTOMATION <<<")
    generate_distribution_plots(df)
    generate_correlation_matrix(df)
    generate_channel_breakdown_plot(df)
    generate_reason_resolution_plot(df)
    generate_time_trends_plot(df)
    generate_outlier_plots(df)
    summary_dict = generate_summary_tables(df)
    print("  [EDA COMPLETE] All analytical plots and tabular breakdowns successfully generated.")
    return summary_dict


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    df_data, _ = run_module1()
    run_module3(df_data)
