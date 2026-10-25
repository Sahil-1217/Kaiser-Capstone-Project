"""
Module 1: Data Integration & Schema Integrity (Phase 1)
Ingests all 5 source CSVs, verifies referential integrity, builds the integrated
denormalized staging dataset, and computes baseline operational metrics.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.config import BASE_DIR, DATA_RAW_DIR, INPUT_FILES


def load_raw_datasets(data_dir: Path = None):
    """
    Ingest all 5 CSV files from the specified data directory or root workspace.
    """
    search_dirs = [data_dir, BASE_DIR, DATA_RAW_DIR]
    search_dirs = [d for d in search_dirs if d is not None and d.exists()]

    datasets = {}
    for key, filename in INPUT_FILES.items():
        found = False
        for sdir in search_dirs:
            file_path = sdir / filename
            if file_path.exists():
                datasets[key] = pd.read_csv(file_path)
                print(f"  [INGESTION] Loaded {filename} from {sdir.name} ({datasets[key].shape[0]} rows, {datasets[key].shape[1]} cols)")
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Required input dataset '{filename}' not found in any search path: {[str(d) for d in search_dirs]}")

    return datasets


def validate_schema_integrity(datasets: dict) -> dict:
    """
    Perform rigorous validation of primary keys, foreign key referential integrity,
    null rates, and data types.
    """
    validation_report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "table_stats": {}
    }

    df_member = datasets["member_profile"]
    df_history = datasets["interaction_history"]
    df_performance = datasets["service_performance"]
    df_operations = datasets["operations_data"]
    df_transcript = datasets["transcript_data"]

    # 1. Primary Key Uniqueness
    if df_member["Member_ID"].duplicated().any():
        dups = df_member["Member_ID"].duplicated().sum()
        validation_report["errors"].append(f"Member_Profile has {dups} duplicate Member_ID primary keys.")
        validation_report["is_valid"] = False

    for name, df in [("Interaction_History", df_history),
                     ("Service_Performance", df_performance),
                     ("Operations_Data", df_operations),
                     ("Transcript_Data", df_transcript)]:
        if df["Interaction_ID"].duplicated().any():
            dups = df["Interaction_ID"].duplicated().sum()
            validation_report["errors"].append(f"{name} has {dups} duplicate Interaction_ID keys.")
            validation_report["is_valid"] = False

    # 2. Foreign Key Integrity: Member_ID
    history_members = set(df_history["Member_ID"])
    profile_members = set(df_member["Member_ID"])
    orphan_members = history_members - profile_members
    if orphan_members:
        validation_report["errors"].append(f"Found {len(orphan_members)} orphan Member_IDs in Interaction_History not in Member_Profile.")
        validation_report["is_valid"] = False
    else:
        print("  [INTEGRITY] Foreign Key check passed: All Member_IDs in Interaction_History exist in Member_Profile.")

    # 3. Foreign Key Integrity: Interaction_ID across all 4 interaction tables
    base_interactions = set(df_history["Interaction_ID"])
    for name, df in [("Service_Performance", df_performance),
                     ("Operations_Data", df_operations),
                     ("Transcript_Data", df_transcript)]:
        curr_interactions = set(df["Interaction_ID"])
        diff_missing = base_interactions - curr_interactions
        diff_extra = curr_interactions - base_interactions
        if diff_missing:
            validation_report["errors"].append(f"{name} is missing {len(diff_missing)} Interaction_IDs present in Interaction_History.")
            validation_report["is_valid"] = False
        if diff_extra:
            validation_report["errors"].append(f"{name} contains {len(diff_extra)} orphan Interaction_IDs not in Interaction_History.")
            validation_report["is_valid"] = False

    if not validation_report["errors"]:
        print("  [INTEGRITY] Foreign Key check passed: 100% referential integrity across all 4 Interaction_ID tables.")

    # 4. Null value check
    for name, df in [("Member_Profile", df_member),
                     ("Interaction_History", df_history),
                     ("Service_Performance", df_performance),
                     ("Operations_Data", df_operations),
                     ("Transcript_Data", df_transcript)]:
        nulls = df.isnull().sum()
        total_nulls = int(nulls.sum())
        validation_report["table_stats"][name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "total_nulls": total_nulls
        }
        if total_nulls > 0:
            validation_report["warnings"].append(f"{name} has {total_nulls} null entries: {nulls[nulls > 0].to_dict()}")

    return validation_report


def build_integrated_dataset(datasets: dict) -> pd.DataFrame:
    """
    Create a unified denormalized staging dataframe joining all 5 sources.
    """
    df_history = datasets["interaction_history"]
    df_member = datasets["member_profile"]
    df_performance = datasets["service_performance"]
    df_operations = datasets["operations_data"]
    df_transcript = datasets["transcript_data"]

    # Sequential merges with explicit validation
    df_merged = df_history.merge(df_member, on="Member_ID", how="inner")
    df_merged = df_merged.merge(df_performance, on="Interaction_ID", how="inner")
    df_merged = df_merged.merge(df_operations, on="Interaction_ID", how="inner")
    df_merged = df_merged.merge(df_transcript, on="Interaction_ID", how="inner")

    print(f"  [INTEGRATION] Successfully joined 5 sources into denormalized staging dataframe: {df_merged.shape[0]} rows, {df_merged.shape[1]} columns.")
    return df_merged


def compute_baseline_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate baseline operational metrics: total volume by channel and region,
    overall escalation rate, FCR rate, mean resolution time, and CSAT.
    """
    total_volume = len(df)
    volume_by_channel = df["Contact_Channel"].value_counts().to_dict()
    volume_by_region = df["Region"].value_counts().to_dict()
    
    escalation_rate = float(df["Escalation_Flag"].mean())
    fcr_rate = float(df["First_Contact_Resolution"].mean())
    mean_resolution = float(df["Resolution_Time"].mean())
    median_resolution = float(df["Resolution_Time"].median())
    mean_csat = float(df["Satisfaction_Score"].mean())

    metrics = {
        "Total_Interactions": total_volume,
        "Unique_Members": int(df["Member_ID"].nunique()),
        "Unique_Agents": int(df["Agent_ID"].nunique()),
        "Overall_Escalation_Rate": round(escalation_rate, 4),
        "Overall_FCR_Rate": round(fcr_rate, 4),
        "Mean_Resolution_Time_Mins": round(mean_resolution, 2),
        "Median_Resolution_Time_Mins": round(median_resolution, 2),
        "Mean_CSAT_Score": round(mean_csat, 2),
        "Volume_By_Channel": volume_by_channel,
        "Volume_By_Region": volume_by_region
    }

    print("\n" + "="*70)
    print(" KAISER PERMANENTE CONTACT CENTER BASELINE OPERATIONAL METRICS")
    print("="*70)
    print(f"  - Total Interaction Volume:    {metrics['Total_Interactions']:,}")
    print(f"  - Unique Active Members:       {metrics['Unique_Members']:,}")
    print(f"  - Unique Agents:               {metrics['Unique_Agents']}")
    print(f"  - Overall Escalation Rate:     {metrics['Overall_Escalation_Rate']*100:.2f}%")
    print(f"  - First Contact Resolution:    {metrics['Overall_FCR_Rate']*100:.2f}%")
    print(f"  - Mean Resolution Time:        {metrics['Mean_Resolution_Time_Mins']} min (Median: {metrics['Median_Resolution_Time_Mins']} min)")
    print(f"  - Mean Satisfaction (CSAT):    {metrics['Mean_CSAT_Score']} / 5.0")
    print("  - Volume Breakdown by Channel:")
    for ch, count in volume_by_channel.items():
        print(f"      * {ch:<22}: {count:>5} ({count/total_volume*100:5.1f}%)")
    print("  - Volume Breakdown by Region:")
    for reg, count in volume_by_region.items():
        print(f"      * {reg:<22}: {count:>5} ({count/total_volume*100:5.1f}%)")
    print("="*70 + "\n")

    return metrics


def run_module1():
    """Execution entry point for Module 1."""
    print("\n>>> EXECUTING MODULE 1: DATA INTEGRATION & SCHEMA INTEGRITY <<<")
    datasets = load_raw_datasets()
    validation = validate_schema_integrity(datasets)
    if not validation["is_valid"]:
        raise ValueError(f"Schema integrity validation failed: {validation['errors']}")
    
    df_integrated = build_integrated_dataset(datasets)
    metrics = compute_baseline_metrics(df_integrated)
    return df_integrated, metrics


if __name__ == "__main__":
    run_module1()
