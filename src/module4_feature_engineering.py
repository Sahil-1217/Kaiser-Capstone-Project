"""
Module 4: Operational Feature Engineering (Phase 4)
Engineers operational, behavioral, and friction features with strict leakage prevention:
1. Contact_Frequency_Score
2. Digital_Adoption_Score
3. Repeat_Contact_Rate
4. Service_Friction_Index
5. Resolution_Complexity_Index
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.config import DIGITAL_CHANNELS, DEPARTMENT_COMPLEXITY_TIERS, FRICTION_WEIGHTS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers member-level and interaction-level features.
    """
    df = df.copy()

    # 1. Member-level behavioral aggregations (computed across interactions per member)
    member_stats = df.groupby("Member_ID").agg(
        Member_Interaction_Count=("Interaction_ID", "count"),
        Member_Digital_Count=("Contact_Channel", lambda ch: ch.isin(DIGITAL_CHANNELS).sum()),
        Member_Repeat_Count=("Total_Contacts", lambda tc: (tc > 1).sum()),
        Member_Mean_Contacts=("Total_Contacts", "mean")
    ).reset_index()

    # Feature 1: Contact_Frequency_Score (Normalized interaction frequency per member)
    max_interactions = member_stats["Member_Interaction_Count"].max()
    member_stats["Contact_Frequency_Score"] = (
        member_stats["Member_Interaction_Count"] / max_interactions
    ).round(4)

    # Feature 2: Digital_Adoption_Score (Proportion of interactions via Chat or Mobile App)
    member_stats["Digital_Adoption_Score"] = (
        member_stats["Member_Digital_Count"] / member_stats["Member_Interaction_Count"]
    ).round(4)

    # Feature 3: Repeat_Contact_Rate (Proportion of member interactions requiring > 1 contact)
    member_stats["Repeat_Contact_Rate"] = (
        member_stats["Member_Repeat_Count"] / member_stats["Member_Interaction_Count"]
    ).round(4)

    # Merge member features back to interaction records
    df = df.merge(
        member_stats[[
            "Member_ID", "Contact_Frequency_Score",
            "Digital_Adoption_Score", "Repeat_Contact_Rate",
            "Member_Interaction_Count"
        ]],
        on="Member_ID",
        how="left"
    )

    # 4. Service_Friction_Index (Composite metric: 0.35 * NormResolution + 0.35 * NormContacts + 0.30 * NonFCR)
    # Clip resolution time at 99th percentile for robust normalization
    rt_p99 = df["Resolution_Time"].quantile(0.99)
    norm_resolution = np.clip(df["Resolution_Time"] / rt_p99, 0.0, 1.0)

    # Max contacts normalization
    max_tc = df["Total_Contacts"].max()
    norm_contacts = df["Total_Contacts"] / max_tc

    # Non-FCR penalty (1 - FCR)
    non_fcr_penalty = 1.0 - df["First_Contact_Resolution"]

    w_res = FRICTION_WEIGHTS["resolution"]
    w_con = FRICTION_WEIGHTS["contacts"]
    w_fcr = FRICTION_WEIGHTS["non_fcr"]

    df["Service_Friction_Index"] = (
        w_res * norm_resolution +
        w_con * norm_contacts +
        w_fcr * non_fcr_penalty
    ).round(4)

    # 5. Resolution_Complexity_Index (Department difficulty tier * duration complexity)
    dept_weights = df["Department"].map(DEPARTMENT_COMPLEXITY_TIERS).fillna(0.50)
    df["Resolution_Complexity_Index"] = (
        dept_weights * (0.40 + 0.60 * norm_resolution)
    ).round(4)

    print("  [FEATURE ENGINEERING] Successfully generated features:")
    print("    - Contact_Frequency_Score       (Range: [0.0, 1.0])")
    print("    - Digital_Adoption_Score        (Range: [0.0, 1.0])")
    print("    - Repeat_Contact_Rate           (Range: [0.0, 1.0])")
    print(f"    - Service_Friction_Index        (Mean: {df['Service_Friction_Index'].mean():.4f}, Max: {df['Service_Friction_Index'].max():.4f})")
    print(f"    - Resolution_Complexity_Index   (Mean: {df['Resolution_Complexity_Index'].mean():.4f}, Max: {df['Resolution_Complexity_Index'].max():.4f})")

    return df


def run_module4(df: pd.DataFrame) -> pd.DataFrame:
    """Execution entry point for Module 4."""
    print("\n>>> EXECUTING MODULE 4: OPERATIONAL FEATURE ENGINEERING <<<")
    df_engineered = engineer_features(df)
    return df_engineered


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    df_data, _ = run_module1()
    df_feat = run_module4(df_data)
