"""
Kaiser Permanente Member Experience Analytics & Intelligence Pipeline
Configuration and Global Settings
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

# Ensure all directories exist
for p in [DATA_RAW_DIR, DATA_PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR, MODELS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Input File Names (Expected in working directory or data/raw)
INPUT_FILES = {
    "member_profile": "Member_Profile.csv",
    "interaction_history": "Interaction_History.csv",
    "service_performance": "Service_Performance.csv",
    "operations_data": "Operations_Data.csv",
    "transcript_data": "Transcript_Data.csv",
}

# Star Schema Export File Names
OUTPUT_FILES = {
    "dim_member": "Dim_Member_Enriched.csv",
    "dim_agent": "Dim_Agent.csv",
    "dim_date": "Dim_Date.csv",
    "fact_interactions": "Fact_Interactions_Enriched.csv",
    "rag_corpus": "RAG_Corpus_Staging.csv",
}

# Kaiser Permanente Operational Domain Values
REGIONS = [
    "Northern California",
    "Southern California",
    "Colorado",
    "Northwest",
    "Mid-Atlantic",
    "Georgia",
    "Hawaii"
]

PLAN_TYPES = [
    "Commercial HMO",
    "Medicare Advantage",
    "Deductible HMO",
    "Point of Service (POS)",
    "Medicaid Managed Care"
]

CONTACT_CHANNELS = [
    "Phone",
    "Chat",
    "Mobile App",
    "Email / Member Portal"
]

DIGITAL_CHANNELS = ["Chat", "Mobile App"]

INTERACTION_REASONS = [
    "Appointments & Scheduling",
    "Billing & Co-Pay Dispute",
    "Claims Processing & Reimbursement",
    "Pharmacy Refill & Prescription Delivery",
    "Provider Network & Specialist Access",
    "Prior Authorization & Referral",
    "Benefits & Coverage Inquiry"
]

DEPARTMENTS = [
    "Appointments",
    "Billing & Claims",
    "Pharmacy Services",
    "Clinical Advice / Care Navigation",
    "Member Services"
]

QUEUES = [
    "General Member Inquiries",
    "Tier 2 Escalations & Grievance",
    "Pharmacy Fulfillment Desk",
    "Specialist Scheduling Desk",
    "Claims Adjudication Queue"
]

SHIFTS = ["Morning", "Day", "Evening", "Night"]

# Department Difficulty Tiers for Resolution_Complexity_Index
DEPARTMENT_COMPLEXITY_TIERS = {
    "Billing & Claims": 1.0,
    "Clinical Advice / Care Navigation": 0.85,
    "Pharmacy Services": 0.65,
    "Member Services": 0.45,
    "Appointments": 0.30
}

# 5 Core Kaiser Operational Themes for NLP
NLP_THEMES = [
    "Appointments",
    "Billing",
    "Claims",
    "Pharmacy",
    "Provider Access"
]

# Service Friction Index Formula Weights
FRICTION_WEIGHTS = {
    "resolution": 0.35,
    "contacts": 0.35,
    "non_fcr": 0.30
}
