# Kaiser Permanente Member Experience Intelligence Platform
## Star Schema Relationship & Data Modeling Specification (Phase 10)

This document specifies the exact Star Schema entity-relationship architecture designed for the Kaiser Permanente Member Experience platform, ensuring optimal query performance, deterministic filter propagation, and ambiguity-free DAX measures in Power BI Desktop.

---

## 1. Star Schema Architecture Overview

```
       +-------------------------------+
       |      Dim_Member_Enriched      |
       |  PK: Member_ID (1)            |
       |  Age, Plan_Type, Region       |
       |  Member_Segment, Friction     |
       +---------------+---------------+
                       | 1
                       |
                       | *
+----------------------+----------------------+----------------------+
|                      Fact_Interactions_Enriched                    |
|  PK: Interaction_ID                                                |
|  FK: Member_ID, Agent_ID, Date_Key                                  |
|  Metrics: Resolution_Time, Satisfaction_Score, Escalation_Flag     |
|  AI Features: Service_Friction_Index, Predicted_Escalation_Prob    |
|  NLP Outputs: Dominant_Topic, Sentiment_Score                      |
+----------------------+----------------------+----------------------+
                       | *                     | *
                       |                       |
                       | 1                     | 1
       +---------------+---------------+       +---------------+---------------+
       |           Dim_Agent           |       |           Dim_Date            |
       |  PK: Agent_ID (1)             |       |  PK: Date_Key (1)             |
       |  Department, Queue, Shift     |       |  Date, Year, Quarter, Month   |
       |  Agent_FCR_Rate, Agent_CSAT   |       |  Day, Day_Name, Week_Number   |
       +-------------------------------+       +-------------------------------+
```

---

## 2. Definitive Entity Relationship Matrix

| Dimension Table (From) | Primary Key | Fact Table (To) | Foreign Key | Cardinality | Cross Filter Direction | Relationship Status | DirectQuery Optimized Index |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Dim_Member_Enriched`** | `Member_ID` | **`Fact_Interactions_Enriched`** | `Member_ID` | **1 to Many (`1:*`)** | **Single** (`Dim` filters `Fact`) | **Active** | `idx_fact_member_id` |
| **`Dim_Agent`** | `Agent_ID` | **`Fact_Interactions_Enriched`** | `Agent_ID` | **1 to Many (`1:*`)** | **Single** (`Dim` filters `Fact`) | **Active** | `idx_fact_agent_id` |
| **`Dim_Date`** | `Date_Key` *(or `Date`)* | **`Fact_Interactions_Enriched`** | `Date_Key` *(or `Interaction_Date`)* | **1 to Many (`1:*`)** | **Single** (`Dim` filters `Fact`) | **Active** | `idx_fact_date_key` |

---

## 3. Table Granularity & Schema Definitions

### 1. `Fact_Interactions_Enriched` (Central Transactional Fact)
- **Granularity:** One record per contact center interaction (Phone, Chat, Mobile App).
- **Primary Key:** `Interaction_ID` (e.g., `KP-INT-20240001`).
- **Foreign Keys:**
  - `Member_ID` $\to$ connects to `Dim_Member_Enriched[Member_ID]`
  - `Agent_ID` $\to$ connects to `Dim_Agent[Agent_ID]`
  - `Date_Key` $\to$ connects to `Dim_Date[Date_Key]`
- **Pre-Interaction Predictors:** `Age`, `Plan_Type`, `Region`, `Contact_Channel`, `Queue`, `Department`, `Shift`, `Total_Contacts`.
- **Engineered & AI Outputs:**
  - `Service_Friction_Index`: Continuous composite friction metric $\in [0, 1]$.
  - `Resolution_Complexity_Index`: Interaction complexity scaling duration by department tier.
  - `Predicted_Escalation_Prob`: Pre-interaction logistic regression predicted probability $\in [0, 1]$.
  - `Predicted_Escalation_Class`: Binary escalation alert flag (1 if prob $\ge 0.50$, else 0).
  - `Dominant_Topic`: Unsupervised LDA operational theme (Appointments, Pharmacy, Claims, Billing, Provider Access).
  - `Sentiment_Score`: PyTorch deep learning continuous sentiment score $\in [0, 1]$.

### 2. `Dim_Member_Enriched` (Member Dimension)
- **Granularity:** One record per unique member (2,621 members).
- **Primary Key:** `Member_ID` (e.g., `KP-MEM-10001`).
- **Enriched Behavioral Personas:**
  - `Member_Segment`: K-Means clustering persona (*Escalation-Prone / High-Friction*, *Digital-First Self-Service*, *Service-Intensive / High-Touch*, *Routine Low-Need Care Seekers*).
  - `Digital_Adoption_Score`: Lifetime proportion of interactions conducted digitally.
  - `Total_Lifetime_Interactions`: Member historical contact volume.
  - `Member_Escalation_Rate`: Lifetime percentage of interactions requiring tier 2 escalation.

### 3. `Dim_Agent` (Agent Dimension)
- **Granularity:** One record per unique customer service / care navigation agent (250 agents).
- **Primary Key:** `Agent_ID` (e.g., `KP-AGT-101`).
- **Operational Profile:** `Department`, `Queue`, `Shift`, `Total_Interactions_Handled`.
- **Performance Benchmarks:** `Agent_FCR_Rate`, `Agent_Avg_CSAT`, `Agent_Escalation_Rate`.

### 4. `Dim_Date` (Calendar Dimension)
- **Granularity:** One record per calendar day (365 continuous days covering 2024-01-01 to 2024-12-31).
- **Primary Key:** `Date_Key` / `Date` (Format: `YYYY-MM-DD`).
- **Hierarchical Attributes:** `Year`, `Quarter` (e.g., `Q1`), `Month` (1–12), `Month_Name` (e.g., `January`), `Day` (1–31), `Day_Name` (e.g., `Monday`), `Week_Number` (1–52), `Is_Weekend` (0 or 1).

### 5. `RAG_Corpus_Staging` (Transcript Knowledge Base)
- **Granularity:** One record per interaction transcript (10,000 interactions).
- **Join Key:** `Interaction_ID` (1:1 relationship with `Fact_Interactions_Enriched` for deep-dive transcript exploration).
- **Payload Attributes:** `Cleaned_Transcript`, `Topic`, `Channel`, `Department`, `Sentiment_Score`, `Metadata_Tags`.

---

## 4. Modeling Rules & Anti-Pattern Protections

1. **Strictly Avoid Bi-Directional Cross-Filtering:**
   - Setting relationships to "Both" causes ambiguous filter propagation paths and degrades DAX measure performance.
   - All relationships are configured as **Single Direction (`Dim` $\to$ `Fact`)**.
2. **Never Join Dimension to Dimension Directly:**
   - E.g., `Dim_Member_Enriched` must never connect directly to `Dim_Agent`. All dimensional correlation occurs through the central `Fact_Interactions_Enriched` table.
3. **No Snowflake Inefficiency:**
   - All member demographic and behavioral persona columns are denormalized into `Dim_Member_Enriched`, eliminating multi-hop joins.
4. **Referential Integrity Validation:**
   - Verified 100% referential integrity in Module 1: every `Member_ID`, `Agent_ID`, and `Interaction_Date` in the Fact table exists within its corresponding Dimension table.
