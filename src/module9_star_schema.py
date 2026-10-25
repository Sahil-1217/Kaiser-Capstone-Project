"""
Module 9: Power BI Star Schema Export Layer (Phase 10)
Structures, validates, and exports clean Star Schema dimension and fact tables
optimized for Power BI ingestion and RAG corpus integration:
1. Dim_Member_Enriched.csv
2. Dim_Agent.csv
3. Dim_Date.csv
4. Fact_Interactions_Enriched.csv
5. RAG_Corpus_Staging.csv
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import DATA_PROCESSED_DIR, BASE_DIR, OUTPUT_FILES


def create_dim_member(df_interactions: pd.DataFrame, df_member_enriched: pd.DataFrame = None) -> pd.DataFrame:
    """
    Creates Dim_Member_Enriched table containing:
    Member_ID, Age, Plan_Type, Region, Member_Segment, Digital_Adoption_Score,
    Total_Lifetime_Interactions, Avg_Member_Friction
    """
    if df_member_enriched is not None and "Member_Segment" in df_member_enriched.columns:
        dim_member = df_member_enriched.copy()
    else:
        # Fallback aggregation
        dim_member = df_interactions.groupby("Member_ID").agg(
            Age=("Age", "first"),
            Plan_Type=("Plan_Type", "first"),
            Region=("Region", "first"),
            Member_Segment=("Member_Segment", "first"),
            Digital_Adoption_Score=("Digital_Adoption_Score", "first"),
            Total_Lifetime_Interactions=("Interaction_ID", "count"),
            Avg_Member_Friction=("Service_Friction_Index", "mean")
        ).reset_index()

    # Select and order required columns
    cols = [
        "Member_ID", "Age", "Plan_Type", "Region",
        "Member_Segment", "Digital_Adoption_Score"
    ]
    # Add extra analytical attributes if available
    for extra in ["Total_Lifetime_Interactions", "Avg_Member_Friction", "Member_Escalation_Rate"]:
        if extra in dim_member.columns and extra not in cols:
            cols.append(extra)

    dim_member = dim_member[cols].drop_duplicates(subset=["Member_ID"]).sort_values("Member_ID")
    return dim_member


def create_dim_agent(df_interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Dim_Agent table containing:
    Agent_ID, Queue, Department, Shift, Total_Interactions_Handled,
    Agent_FCR_Rate, Agent_Avg_CSAT
    """
    dim_agent = df_interactions.groupby("Agent_ID").agg(
        Department=("Department", lambda x: x.mode()[0] if not x.empty else "General"),
        Queue=("Queue", lambda x: x.mode()[0] if not x.empty else "General"),
        Shift=("Shift", lambda x: x.mode()[0] if not x.empty else "Day"),
        Total_Interactions_Handled=("Interaction_ID", "count"),
        Agent_FCR_Rate=("First_Contact_Resolution", "mean"),
        Agent_Avg_CSAT=("Satisfaction_Score", "mean"),
        Agent_Escalation_Rate=("Escalation_Flag", "mean")
    ).reset_index()

    dim_agent["Agent_FCR_Rate"] = dim_agent["Agent_FCR_Rate"].round(4)
    dim_agent["Agent_Avg_CSAT"] = dim_agent["Agent_Avg_CSAT"].round(2)
    dim_agent["Agent_Escalation_Rate"] = dim_agent["Agent_Escalation_Rate"].round(4)

    return dim_agent.sort_values("Agent_ID")


def create_dim_date(df_interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a continuous calendar dimension table covering all interaction dates:
    Date, Year, Month, Month_Name, Quarter, Day, Day_Name, Week_Number, Is_Weekend
    """
    dates = pd.to_datetime(df_interactions["Interaction_Date"]).dropna()
    min_date = dates.min()
    max_date = dates.max()

    calendar_range = pd.date_range(start=min_date, end=max_date, freq="D")
    
    dim_date = pd.DataFrame({
        "Date": calendar_range.strftime("%Y-%m-%d"),
        "Date_Key": calendar_range.strftime("%Y-%m-%d"),
        "Year": calendar_range.year,
        "Quarter": [f"Q{q}" for q in calendar_range.quarter],
        "Month": calendar_range.month,
        "Month_Name": calendar_range.strftime("%B"),
        "Day": calendar_range.day,
        "Day_Name": calendar_range.strftime("%A"),
        "Week_Number": calendar_range.isocalendar().week.astype(int),
        "Is_Weekend": (calendar_range.dayofweek >= 5).astype(int)
    })

    return dim_date


def create_fact_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates Fact_Interactions_Enriched table containing:
    All original transactional metrics +
    Service_Friction_Index, Resolution_Complexity_Index,
    Predicted_Escalation_Prob, Predicted_Escalation_Class,
    Dominant_Topic, Sentiment_Score, Date_Key
    """
    df_copy = df.copy()
    if "Predicted_Escalation_Class" not in df_copy.columns and "Predicted_Escalation_Prob" in df_copy.columns:
        df_copy["Predicted_Escalation_Class"] = (df_copy["Predicted_Escalation_Prob"] >= 0.50).astype(int)
    if "Date_Key" not in df_copy.columns and "Interaction_Date" in df_copy.columns:
        df_copy["Date_Key"] = pd.to_datetime(df_copy["Interaction_Date"]).dt.strftime("%Y-%m-%d")

    fact_cols = [
        "Interaction_ID",
        "Member_ID",
        "Agent_ID",
        "Interaction_Date",
        "Date_Key",
        "Contact_Channel",
        "Interaction_Reason",
        "Queue",
        "Department",
        "Shift",
        "Total_Contacts",
        "Resolution_Time",
        "First_Contact_Resolution",
        "Satisfaction_Score",
        "Escalation_Flag",
        "Service_Friction_Index",
        "Resolution_Complexity_Index",
        "Predicted_Escalation_Prob",
        "Predicted_Escalation_Class",
        "Dominant_Topic",
        "Sentiment_Score"
    ]
    
    available_cols = [c for c in fact_cols if c in df_copy.columns]
    return df_copy[available_cols].copy()


def create_rag_corpus_staging(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates clean, embeddings-ready transcript metadata table:
    Interaction_ID, Member_ID, Topic, Channel, Department, Sentiment_Score,
    Escalation_Flag, Cleaned_Transcript, Metadata_Tags
    """
    rag_df = pd.DataFrame({
        "Interaction_ID": df["Interaction_ID"],
        "Topic": df.get("Dominant_Topic", "General Inquiries"),
        "Channel": df["Contact_Channel"],
        "Cleaned_Transcript": df.get("Cleaned_Transcript", df["Call_Transcript"].fillna("") + " " + df["Chat_Transcript"].fillna("")),
        "Department": df["Department"],
        "Sentiment_Score": df.get("Sentiment_Score", 0.5),
        "Escalation_Flag": df["Escalation_Flag"]
    })

    # Metadata tags string formatted for vector DB retrieval indexing
    rag_df["Metadata_Tags"] = (
        "topic:" + rag_df["Topic"].astype(str) + " | " +
        "channel:" + rag_df["Channel"].astype(str) + " | " +
        "dept:" + rag_df["Department"].astype(str) + " | " +
        "escalated:" + rag_df["Escalation_Flag"].astype(str)
    )

    return rag_df


def export_star_schema_tables(tables: dict):
    """
    Exports all Star Schema tables to both data/processed/ and root workspace.
    """
    print("\n  [EXPORTING STAR SCHEMA & RAG STAGING]")
    for key, df in tables.items():
        filename = OUTPUT_FILES[key]
        
        # Save to data/processed
        processed_path = DATA_PROCESSED_DIR / filename
        df.to_csv(processed_path, index=False)

        # Save to root workspace for immediate Power BI file loading
        root_path = BASE_DIR / filename
        df.to_csv(root_path, index=False)

        print(f"    * Exported {filename:<28} -> {df.shape[0]:>6} rows, {df.shape[1]:>2} cols")


def run_module9(df_enriched: pd.DataFrame, df_member_enriched: pd.DataFrame = None) -> dict:
    """Execution entry point for Module 9."""
    print("\n>>> EXECUTING MODULE 9: POWER BI STAR SCHEMA EXPORT LAYER <<<")
    
    dim_member = create_dim_member(df_enriched, df_member_enriched)
    dim_agent = create_dim_agent(df_enriched)
    dim_date = create_dim_date(df_enriched)
    fact_interactions = create_fact_interactions(df_enriched)
    rag_corpus = create_rag_corpus_staging(df_enriched)

    star_schema_tables = {
        "dim_member": dim_member,
        "dim_agent": dim_agent,
        "dim_date": dim_date,
        "fact_interactions": fact_interactions,
        "rag_corpus": rag_corpus
    }

    export_star_schema_tables(star_schema_tables)
    print("  [STAR SCHEMA EXPORT COMPLETE] All Power BI and RAG artifacts successfully saved.")
    return star_schema_tables


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    from src.module4_feature_engineering import run_module4
    from src.module5_escalation_ml import run_module5
    from src.module6_segmentation import run_module6
    from src.module7_sentiment_dl import run_module7
    from src.module8_nlp_topics import run_module8

    df_data, _ = run_module1()
    df_feat = run_module4(df_data)
    df_scored, _, _ = run_module5(df_feat)
    df_seg, df_mem, _ = run_module6(df_scored)
    df_sent, _ = run_module7(df_seg)
    df_final, _, _ = run_module8(df_sent)
    tables = run_module9(df_final, df_mem)
