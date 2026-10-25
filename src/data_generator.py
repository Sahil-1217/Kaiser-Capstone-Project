"""
Kaiser Permanente Analytics Pipeline - Realistic Synthetic Data Generator
Generates statistically calibrated datasets matching the exact schema of:
1. Member_Profile.csv
2. Interaction_History.csv
3. Service_Performance.csv
4. Operations_Data.csv
5. Transcript_Data.csv
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import (
    RANDOM_STATE, REGIONS, PLAN_TYPES, CONTACT_CHANNELS, DIGITAL_CHANNELS,
    INTERACTION_REASONS, DEPARTMENTS, QUEUES, SHIFTS, NLP_THEMES,
    DATA_RAW_DIR, BASE_DIR
)

# Fix seeds
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# Transcript templates per theme
TRANSCRIPT_TEMPLATES = {
    "Appointments": {
        "call": [
            "Member: Hi, I've been trying to book an appointment with my primary care physician Dr. Martinez in Oakland, but the portal says no slots for 3 weeks. Agent: I understand your concern. Let me check the clinical schedule and urgent appointment openings. I found an opening this Thursday at 10:15 AM at the Medical Center.",
            "Member: I need to reschedule my endocrinology follow-up visit because of a work conflict. Agent: Certainly, let's see available slots for specialist appointments next week. We have Tuesday morning or Friday afternoon with Dr. Patel.",
            "Member: Hello, my child has a persistent earache and I need a same-day pediatric appointment. Agent: I can connect you immediately to our pediatric care triage desk. We have a walk-in slot at 2:30 PM."
        ],
        "chat": [
            "User: Need to schedule routine annual physical. Bot: Checking primary care slots. Connecting to scheduling agent. Agent: Hello! I can confirm Dr. Wong has an opening on November 14 at 9:00 AM. Would that work?",
            "User: Can I get a telehealth video visit today for sinus pressure? Agent: Yes, our virtual care clinic has an immediate appointment in 20 minutes. I have reserved it for you."
        ],
        "feedback_pos": [
            "Nurse and agent were so prompt in finding a same-day pediatric slot for my daughter. Excellent service!",
            "Effortless appointment booking through the app. The receptionist was very courteous.",
            "Dr. Davis was thorough and the scheduling team accommodated my busy work schedule."
        ],
        "feedback_neg": [
            "Took over 3 weeks just to get a routine primary care slot. Kaiser needs more doctors.",
            "Appointment was cancelled last minute and rescheduled two months out without explanation.",
            "Long hold time on the phone just to be told no specialist appointments are open."
        ]
    },
    "Billing": {
        "call": [
            "Member: I received an unexpected bill for $385 from my visit last month. My plan says office visits have a $30 copay! Agent: Let me pull up your account and claims ledger. It looks like laboratory diagnostic blood work was billed under your annual deductible rather than the copay tier.",
            "Member: You charged my credit card twice for the same outpatient physical therapy co-pay. Agent: I apologize for the processing error. I have processed an immediate credit adjustment for $45 back to your card.",
            "Member: I need clarification on my out-of-pocket maximum progress for 2024. Agent: Reviewing your plan summary. You have met $2,850 of your $4,000 annual out-of-pocket limit."
        ],
        "chat": [
            "User: Why is there an outstanding balance on my statement when I paid at the clinic? Agent: Looking into the ledger. The payment cleared but the billing system had not matched the invoice number. Corrected now.",
            "User: Need payment plan setup for deductible expenses from surgery. Agent: I can enroll you into a 6-month zero-interest payment arrangement of $120 per month."
        ],
        "feedback_pos": [
            "The billing representative clearly explained my deductible and corrected the duplicate charge immediately.",
            "Very transparent explanation of my co-insurance and out-of-pocket limits.",
            "Quick resolution of my billing statement inquiry via chat."
        ],
        "feedback_neg": [
            "Received an incorrect bill with hidden charges. Customer service gave me conflicting answers.",
            "Charged twice for my co-pay and still waiting 14 business days for a refund.",
            "Confusing billing statement with zero breakdown of what Kaiser covered vs deductible."
        ]
    },
    "Claims": {
        "call": [
            "Member: I had an emergency room visit while traveling out of state and Kaiser denied my claim as out-of-network! Agent: Let me examine the emergency medical claim submission. Emergency care is covered at in-network rates under prudent layperson guidelines. I am reopening an expedited claim appeal.",
            "Member: I submitted a claim reimbursement form for an urgent care visit six weeks ago and have received no update. Agent: I see the claim was delayed waiting for the itemized provider invoice. I have escalated this to claims adjudication for approval.",
            "Member: My explanation of benefits (EOB) shows an unapplied discount for ambulance transport. Agent: Let me cross-check the claim adjudication details with our third-party billing team."
        ],
        "chat": [
            "User: Status check on Claim #CLM-89210? Agent: The claim was approved yesterday for $720.00 reimbursement. A check has been mailed to your address.",
            "User: Why was my physical therapy claim denied? Agent: It was flagged because the referral authorization expired last Friday. I will request retroactive clinical extension."
        ],
        "feedback_pos": [
            "My out-of-state emergency claim appeal was resolved smoothly thanks to the supervisor's help.",
            "Claim reimbursement was processed and deposited directly into my bank account quickly.",
            "The claims specialist walked me through the entire EOB breakdown with great patience."
        ],
        "feedback_neg": [
            "Emergency claim unfairly denied while traveling. Spent hours fighting bureaucracy.",
            "Claims department took over 60 days to process a simple urgent care reimbursement.",
            "Lost my submitted documentation twice. Unacceptable lack of coordination in claims."
        ]
    },
    "Pharmacy": {
        "call": [
            "Member: My mail-order maintenance prescription for atorvastatin was supposed to arrive Monday, but tracking has not updated. Agent: Checking the fulfillment center. There was a temporary supplier backorder. We can authorize an emergency 30-day supply at your local Kaiser pharmacy today.",
            "Member: The pharmacy tells me my doctor needs to submit a prior authorization for my migraine medication. Agent: I see the clinical requirement. I will send an urgent prior authorization request directly to your neurologist's clinical team.",
            "Member: I need to set up automatic refills and home delivery for my diabetic supplies and insulin. Agent: Glad to assist. I have activated 90-day auto-refill mail order with zero copay."
        ],
        "chat": [
            "User: Can you transfer my prescription to the San Jose Kaiser pharmacy? Agent: Yes! Transfer completed. Your medication will be ready for pickup after 3:00 PM.",
            "User: Need dosage verification on my blood pressure prescription refill. Agent: Connecting you to an on-call clinical pharmacist for consultation."
        ],
        "feedback_pos": [
            "Mail-order pharmacy is a lifesaver. Arrived in 2 days in temperature-controlled packaging.",
            "Pharmacist took the time to explain drug interactions and arranged an emergency local refill.",
            "Seamless refill request through the mobile app. Ready at the counter in 20 minutes."
        ],
        "feedback_neg": [
            "Mail-order prescription delayed 5 days and I ran completely out of blood pressure medicine.",
            "Pharmacy refused to fill without doctor authorization even though it was an ongoing refill.",
            "Stood in line for 45 minutes at the pharmacy counter only to be told it wasn't ready."
        ]
    },
    "Provider Access": {
        "call": [
            "Member: I need a specialist referral to a pediatric dermatologist for my son's eczema, but care navigation told me wait time is 4 months. Agent: Let me explore in-network affiliated community providers or virtual dermatology triage to get him seen within two weeks.",
            "Member: My long-time primary care physician retired and I need help choosing a new doctor who speaks Spanish. Agent: I can help you select a bilingual PCP at the Santa Clara medical offices with great patient reviews.",
            "Member: I need access to a licensed behavioral health therapist for anxiety, but none have open panels. Agent: Kaiser has expanded mental health networks through our Magellan and Beacon partnerships. Let me connect you directly to behavioral care navigation."
        ],
        "chat": [
            "User: How do I find an in-network physical therapist near Pasadena? Agent: Here are 3 Kaiser medical facilities and 2 affiliated outpatient rehab centers with open appointments this week.",
            "User: Need to verify if Dr. Hernandez is accepting new patients. Agent: Yes, Dr. Hernandez has an open panel for adult medicine. I can assign him as your PCP now."
        ],
        "feedback_pos": [
            "Care navigator connected me to a wonderful therapist within 48 hours. So grateful!",
            "Found an amazing bilingual pediatrician for my kids. High quality network of doctors.",
            "The care coordinator followed up personally to ensure my specialist referral went through."
        ],
        "feedback_neg": [
            "Severe shortage of mental health therapists. Wait times for care are unacceptable.",
            "Referral was delayed for weeks between primary care and neurology. Care navigation is broken.",
            "Unable to find an in-network provider within 25 miles of my home."
        ]
    }
}


def generate_synthetic_data(n_members: int = 3500, n_interactions: int = 10000):
    """
    Generate synthetic data for the 5 files with realistic healthcare patterns.
    """
    print(f"[DATA GENERATOR] Generating {n_members} members and {n_interactions} interactions...")

    # 1. Member Profile
    member_ids = [f"KP-MEM-{10000 + i}" for i in range(1, n_members + 1)]
    ages = np.clip(np.random.normal(loc=48.5, scale=16.0, size=n_members).astype(int), 18, 92)
    
    plan_weights = [0.38, 0.26, 0.16, 0.11, 0.09]  # HMO, Medicare, Deductible, POS, Medicaid
    plans = np.random.choice(PLAN_TYPES, size=n_members, p=plan_weights)

    region_weights = [0.35, 0.28, 0.12, 0.09, 0.07, 0.05, 0.04]
    regions = np.random.choice(REGIONS, size=n_members, p=region_weights)

    df_member = pd.DataFrame({
        "Member_ID": member_ids,
        "Age": ages,
        "Plan_Type": plans,
        "Region": regions
    })

    # Precompute member tech propensity based on age
    # Younger members have higher probability of using Chat / Mobile App
    member_tech_affinity = 1.0 / (1.0 + np.exp((ages - 50.0) / 12.0))  # higher for younger
    member_affinity_dict = dict(zip(member_ids, member_tech_affinity))

    # 2. Interaction History
    interaction_ids = [f"KP-INT-{20240001 + i}" for i in range(n_interactions)]
    
    # Sample members with some members having frequent interactions (Pareto principle)
    member_sampling_weights = np.random.exponential(scale=1.0, size=n_members)
    member_sampling_weights /= member_sampling_weights.sum()
    sampled_member_ids = np.random.choice(member_ids, size=n_interactions, p=member_sampling_weights)

    # Interaction Dates spanning 2024
    start_date = datetime(2024, 1, 1)
    date_offsets = np.random.randint(0, 365, size=n_interactions)
    interaction_dates = [(start_date + timedelta(days=int(d))).strftime("%Y-%m-%d") for d in date_offsets]

    # Reasons
    reason_weights = [0.24, 0.20, 0.18, 0.16, 0.10, 0.07, 0.05]
    interaction_reasons = np.random.choice(INTERACTION_REASONS, size=n_interactions, p=reason_weights)

    # Contact Channel depending on member digital affinity
    channels = []
    for m_id in sampled_member_ids:
        affinity = member_affinity_dict[m_id]
        p_phone = max(0.20, 0.65 - 0.40 * affinity)
        p_chat = max(0.10, 0.15 + 0.20 * affinity)
        p_mobile = max(0.08, 0.12 + 0.22 * affinity)
        p_email = max(0.02, 1.0 - (p_phone + p_chat + p_mobile))
        probs = np.array([p_phone, p_chat, p_mobile, p_email])
        probs /= probs.sum()
        ch = np.random.choice(CONTACT_CHANNELS, p=probs)
        channels.append(ch)

    # Total Contacts (repeat interaction count)
    # Higher for complex reasons (Billing, Claims)
    total_contacts_list = []
    for r in interaction_reasons:
        if "Claims" in r or "Billing" in r or "Prior Authorization" in r:
            tc = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.40, 0.26, 0.16, 0.10, 0.05, 0.03])
        else:
            tc = np.random.choice([1, 2, 3, 4, 5, 6], p=[0.72, 0.18, 0.06, 0.02, 0.01, 0.01])
        total_contacts_list.append(tc)

    df_history = pd.DataFrame({
        "Interaction_ID": interaction_ids,
        "Member_ID": sampled_member_ids,
        "Contact_Channel": channels,
        "Interaction_Date": interaction_dates,
        "Interaction_Reason": interaction_reasons,
        "Total_Contacts": total_contacts_list
    })

    # 3. Operations Data
    agents = [f"KP-AGT-{100 + i}" for i in range(1, 251)]
    sampled_agents = np.random.choice(agents, size=n_interactions)

    # Department and Queue mapping
    dept_list = []
    queue_list = []
    for r in interaction_reasons:
        if "Appointment" in r:
            dept = "Appointments"
            q = "Specialist Scheduling Desk"
        elif "Billing" in r or "Claims" in r:
            dept = "Billing & Claims"
            q = "Claims Adjudication Queue" if "Claims" in r else "Tier 2 Escalations & Grievance"
        elif "Pharmacy" in r:
            dept = "Pharmacy Services"
            q = "Pharmacy Fulfillment Desk"
        elif "Provider" in r or "Prior Authorization" in r:
            dept = "Clinical Advice / Care Navigation"
            q = "Tier 2 Escalations & Grievance"
        else:
            dept = "Member Services"
            q = "General Member Inquiries"
        dept_list.append(dept)
        queue_list.append(q)

    shifts = np.random.choice(SHIFTS, size=n_interactions, p=[0.36, 0.42, 0.18, 0.04])

    df_operations = pd.DataFrame({
        "Interaction_ID": interaction_ids,
        "Agent_ID": sampled_agents,
        "Queue": queue_list,
        "Department": dept_list,
        "Shift": shifts
    })

    # 4. Service Performance
    # Resolution Time (log-normal, minutes), longer for claims/billing and high contacts
    base_res_time = np.random.lognormal(mean=2.4, sigma=0.55, size=n_interactions) # median ~11 min
    res_times = []
    for i in range(n_interactions):
        r = interaction_reasons[i]
        tc = total_contacts_list[i]
        ch = channels[i]
        mult = 1.0
        if "Claims" in r or "Billing" in r:
            mult += 0.40
        if tc > 2:
            mult += 0.25 * (tc - 1)
        if ch == "Phone":
            mult += 0.20
        res = np.clip(base_res_time[i] * mult, 2.0, 95.0)
        res_times.append(round(float(res), 2))

    # First Contact Resolution (FCR)
    # Higher for tc == 1, lower for complex reasons
    fcr_list = []
    for i in range(n_interactions):
        tc = total_contacts_list[i]
        if tc == 1:
            p_fcr = 0.88
        elif tc == 2:
            p_fcr = 0.45
        else:
            p_fcr = 0.15
        fcr = int(np.random.rand() < p_fcr)
        fcr_list.append(fcr)

    # Escalation Flag
    # Realistically correlated with Phone channel, multiple contacts, long res time, non-FCR
    escalation_flags = []
    for i in range(n_interactions):
        ch = channels[i]
        tc = total_contacts_list[i]
        rt = res_times[i]
        fcr = fcr_list[i]
        
        # Logistic probability model for escalation
        logit = -2.9
        if ch == "Phone":
            logit += 0.65
        elif ch in DIGITAL_CHANNELS:
            logit -= 0.55
        if tc >= 2:
            logit += 0.60 * (tc - 1)
        if rt > 20.0:
            logit += 0.50
        if fcr == 0:
            logit += 0.70
        
        prob = 1.0 / (1.0 + np.exp(-logit))
        esc = int(np.random.rand() < prob)
        escalation_flags.append(esc)

    # Satisfaction Score (CSAT 1 to 5)
    # Strongly negatively correlated with res time, escalations, repeat contacts
    csat_scores = []
    for i in range(n_interactions):
        esc = escalation_flags[i]
        fcr = fcr_list[i]
        rt = res_times[i]
        tc = total_contacts_list[i]

        score = 4.2
        if esc == 1:
            score -= 1.8
        if fcr == 0:
            score -= 0.8
        if tc > 2:
            score -= 0.3 * (tc - 1)
        if rt > 25.0:
            score -= 0.6
        elif rt < 10.0:
            score += 0.4
        
        # Add slight noise
        score += np.random.normal(0, 0.45)
        final_csat = int(np.clip(round(score), 1, 5))
        csat_scores.append(final_csat)

    df_performance = pd.DataFrame({
        "Interaction_ID": interaction_ids,
        "Resolution_Time": res_times,
        "Escalation_Flag": escalation_flags,
        "Satisfaction_Score": csat_scores,
        "First_Contact_Resolution": fcr_list
    })

    # 5. Transcript Data
    call_transcripts = []
    chat_transcripts = []
    feedback_comments = []

    for i in range(n_interactions):
        r = interaction_reasons[i]
        ch = channels[i]
        esc = escalation_flags[i]
        csat = csat_scores[i]

        # Map reason to one of 5 core themes
        if "Appointment" in r:
            theme = "Appointments"
        elif "Billing" in r:
            theme = "Billing"
        elif "Claims" in r:
            theme = "Claims"
        elif "Pharmacy" in r:
            theme = "Pharmacy"
        else:
            theme = "Provider Access"

        theme_data = TRANSCRIPT_TEMPLATES[theme]

        # Call transcript
        if ch == "Phone":
            call_text = random.choice(theme_data["call"])
            chat_text = ""
        elif ch in DIGITAL_CHANNELS:
            call_text = ""
            chat_text = random.choice(theme_data["chat"])
        else:
            call_text = f"Email inquiry regarding {r}. Member requested review of documentation."
            chat_text = ""

        # Feedback Comments based on CSAT
        if csat >= 4:
            feedback = random.choice(theme_data["feedback_pos"])
        elif csat <= 2:
            feedback = random.choice(theme_data["feedback_neg"])
        else:
            neutral_options = [
                f"The issue with {r.lower()} was eventually handled, but communication could be improved.",
                "Service was acceptable. Hold times were slightly long.",
                "Standard contact center experience. Nothing exceptional."
            ]
            feedback = random.choice(neutral_options)

        call_transcripts.append(call_text)
        chat_transcripts.append(chat_text)
        feedback_comments.append(feedback)

    df_transcript = pd.DataFrame({
        "Interaction_ID": interaction_ids,
        "Call_Transcript": call_transcripts,
        "Chat_Transcript": chat_transcripts,
        "Feedback_Comments": feedback_comments
    })

    # Save CSVs to both data/raw/ and root working directory
    file_dfs = {
        "Member_Profile.csv": df_member,
        "Interaction_History.csv": df_history,
        "Service_Performance.csv": df_performance,
        "Operations_Data.csv": df_operations,
        "Transcript_Data.csv": df_transcript,
    }

    for filename, df in file_dfs.items():
        # Save to data/raw
        raw_path = DATA_RAW_DIR / filename
        df.to_csv(raw_path, index=False)
        # Save to root workspace
        root_path = BASE_DIR / filename
        df.to_csv(root_path, index=False)
        print(f"  [SAVED] {filename}: {df.shape[0]} rows, {df.shape[1]} cols")

    print("[DATA GENERATOR] Successfully generated all 5 datasets.")
    return file_dfs


if __name__ == "__main__":
    generate_synthetic_data()
