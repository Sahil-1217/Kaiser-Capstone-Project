# Kaiser Permanente Member Experience Intelligence Platform
## Power BI Desktop Implementation & Dual-Environment Setup Guide (Phase 10)

This document provides complete instructions for connecting Power BI Desktop (on an office laptop or personal workstation) to the cloud PostgreSQL database (Neon / AWS RDS / local PostgreSQL), configuring Star Schema relationships, writing DAX measures, setting up Row-Level Security (RLS), and building the required **Pages 1 through 6**.

---

## 1. Dual-Environment Database Connection Setup

### Option A: Connecting to Cloud PostgreSQL (Neon / Cloud DB)
1. Open **Power BI Desktop**.
2. Click **Get Data** $\to$ **More...** $\to$ Select **PostgreSQL database** $\to$ Click **Connect**.
3. **Connection Dialog:**
   - **Server:** Enter your cloud host with port (e.g., `ep-xyz.us-east-2.aws.neon.tech:5432` or your PostgreSQL host).
   - **Database:** `neondb` (or `kaiser_analytics`).
   - **Data Connectivity Mode:** Choose **Import** (recommended for fastest responsiveness) or **DirectQuery** (for live real-time queries).
4. **Authentication:**
   - Select **Database** tab on the left.
   - Enter your `User name` and `Password` from your `.env` file.
   - Click **Connect**. If prompted regarding encryption, ensure SSL is enabled.
5. In the Navigator window, select the 4 core Star Schema tables (and optional RAG staging):
   - `fact_interactions_enriched`
   - `dim_member_enriched`
   - `dim_agent`
   - `dim_date`
   - `rag_corpus_staging` (optional for transcript text drilldown)
6. Click **Transform Data** or **Load**.

### Option B: Local CSV Fallback (Zero-Driver Ingestion)
If network firewalls block outbound database port 5432:
1. Copy the 4 CSV files from `data/processed/` to your office machine.
2. In Power BI Desktop: **Get Data** $\to$ **Text/CSV** $\to$ Load `Fact_Interactions_Enriched.csv`, `Dim_Member_Enriched.csv`, `Dim_Agent.csv`, and `Dim_Date.csv`.

---

## 2. Star Schema Relationship Matrix (Model View)

Navigate to **Model View** in Power BI Desktop. Ensure all relationships are configured as **Single Direction (1 to Many)**:

| From Table (1 - Dimension) | Primary Key Column | To Table (* - Fact) | Foreign Key Column | Cardinality | Cross Filter Direction | Security Filtering |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`dim_member_enriched`** | `member_id` | **`fact_interactions_enriched`** | `member_id` | `1 to Many (1:*)` | **Single** (`dim` filters `fact`) | Default |
| **`dim_agent`** | `agent_id` | **`fact_interactions_enriched`** | `agent_id` | `1 to Many (1:*)` | **Single** (`dim` filters `fact`) | Default |
| **`dim_date`** | `date_key` *(or `date`)* | **`fact_interactions_enriched`** | `date_key` *(or `interaction_date`)* | `1 to Many (1:*)` | **Single** (`dim` filters `fact`) | Default |

> **Date Table Configuration:** Right-click `dim_date` $\to$ **Mark as Date Table** $\to$ Select the `date` column.

---

## 3. Production Row-Level Security (RLS) Configuration

Navigate to **Modeling** $\to$ **Manage Roles** $\to$ **Create**:

### Role 1: `Role_Department_Manager`
Restricts visibility so department managers only view interactions within their direct operational purview.
- **Table:** `fact_interactions_enriched`
- **DAX Filter Expression:**
```dax
[department] = 
LOOKUPVALUE(
    user_department_mapping[department],
    user_department_mapping[userprincipalname],
    USERPRINCIPALNAME()
)
```
*(For static testing in Desktop: `[department] = "Billing & Claims"`)*

---

### Role 2: `Role_Regional_Director`
Restricts geographic visibility so regional executives (Northern California, Southern California, Colorado, Mid-Atlantic) only see member cohorts from their region.
- **Table:** `dim_member_enriched`
- **DAX Filter Expression:**
```dax
[region] = "Northern California"
```
*(Dynamic Entra ID expression: `[region] = LOOKUPVALUE(user_region_mapping[region], user_region_mapping[userprincipalname], USERPRINCIPALNAME())`)*

---

## 4. Visual Layout Specifications: Pages 1 to 6

---

### Page 1: Interaction Summary (Executive Contact Center Pulse)
- **Target Audience:** Chief Customer Officer, VP of Health Plan Operations.
- **Top Executive KPI Cards (Row across top):**
  1. Card 1: `[Total Interactions]` (10,000)
  2. Card 2: `[Total Members]` (2,621)
  3. Card 3: `[Escalation Rate %]` (17.38%)
  4. Card 4: `[First Contact Resolution (FCR) Rate %]` (63.47%)
  5. Card 5: `[Average Resolution Time (Mins)]` (17.92 min)
- **Visual 1 (Donut Chart - Channel Volume):**
  - Legend: `fact_interactions_enriched[contact_channel]` (Phone, Chat, Mobile App)
  - Values: `[Total Interactions]`
- **Visual 2 (Clustered Bar Chart - Regional Distribution):**
  - Y-Axis: `dim_member_enriched[region]` (Northern CA, Southern CA, Colorado, Mid-Atlantic)
  - X-Axis: `[Total Interactions]`
  - Tooltip: `[Escalation Rate %]`
- **Visual 3 (Line & Stacked Column Chart - Monthly Trend):**
  - X-Axis: `dim_date[month_name]` (Ordered by `month`)
  - Column Values: `[Total Interactions]`
  - Line Values: `[Escalation Rate %]`
- **Slicers:** Date Range, Member Plan Type, Contact Channel.

---

### Page 2: Satisfaction & Sentiment (Member Experience Diagnostics)
- **Target Audience:** Clinical Care Navigation Leaders, Member Satisfaction Directors.
- **Top KPI Cards:**
  1. Card 1: `[Average CSAT]` (3.46 / 5.0)
  2. Card 2: `[Net Promoter Proxy %]` (Positive vs Negative CSAT ratio)
  3. Card 3: `[Positive Sentiment Rate %]` (58.8%)
  4. Card 4: `[Average Service Friction Index]` (0.31)
- **Visual 1 (Bar Chart - CSAT by Department):**
  - Y-Axis: `fact_interactions_enriched[department]`
  - X-Axis: `[Average CSAT]`
  - Reference Line: Target Goal (4.00 CSAT)
- **Visual 2 (100% Stacked Bar Chart - Sentiment by Channel):**
  - Y-Axis: `fact_interactions_enriched[contact_channel]`
  - X-Axis: `[Positive Sentiment Volume]` vs Negative Volume
- **Visual 3 (Scatter Plot - Friction vs Satisfaction):**
  - X-Axis: `fact_interactions_enriched[service_friction_index]`
  - Y-Axis: `fact_interactions_enriched[satisfaction_score]`
  - Detail: `interaction_id` (or aggregated by `queue`)
  - Trendline: Demonstrates the inverse correlation ($\rho = -0.46$).

---

### Page 3: Escalation Analytics (Operational Risk & Machine Learning)
- **Target Audience:** Contact Center Supervisors, Grievance & Appeals Taskforce.
- **Top KPI Cards:**
  1. Card 1: `[Total Escalations]` (1,738)
  2. Card 2: `[Predicted High Risk Escalations]` (Flags with probability $\ge 0.50$)
  3. Card 3: `[Model Escalation Recall %]` (66.1% captured in advance)
  4. Card 4: `[Average Predicted Escalation Probability]` (0.416)
- **Visual 1 (Clustered Bar Chart - Escalation Disparity by Channel):**
  - Y-Axis: `fact_interactions_enriched[contact_channel]`
  - X-Axis: `[Escalation Rate %]`
  - *Highlight: Phone calls escalate at 25.2% vs Digital at 10.7%.*
- **Visual 2 (Confusion Matrix / Contingency Table):**
  - Rows: `fact_interactions_enriched[escalation_flag]` (Actual Escalated: 0 or 1)
  - Columns: `fact_interactions_enriched[predicted_escalation_class]` (Predicted: 0 or 1)
  - Values: `[Total Interactions]`
- **Visual 3 (Queue Risk Breakdown Table):**
  - Columns: `department`, `queue`, `[Total Interactions]`, `[Escalation Rate %]`, `[Predicted Escalation Exposure %]`
  - Conditional Formatting: Red data bars on High-Risk exposure.

---

### Page 4: Member Segments (K-Means Behavioral Personas)
- **Target Audience:** Population Health, Member Retention Strategy.
- **Persona Metric Cards (4 Segment Cards):**
  1. Cluster 0: *Escalation-Prone / High-Friction* (720 members, 35.8% Escalation, 0.481 Friction)
  2. Cluster 1: *Digital-First Self-Service* (738 members, 6.6% Escalation, 0.239 Friction)
  3. Cluster 2: *Service-Intensive / High-Touch* (528 members, 17.2% Escalation, 0.325 Friction)
  4. Cluster 3: *Routine Low-Need Care Seekers* (635 members, 9.5% Escalation, 0.212 Friction)
- **Visual 1 (Donut Chart - Persona Cohort Distribution):**
  - Legend: `dim_member_enriched[member_segment]`
  - Values: `[Total Members]`
- **Visual 2 (Scatter Plot - Digital Adoption vs Friction):**
  - X-Axis: `dim_member_enriched[digital_adoption_score]`
  - Y-Axis: `dim_member_enriched[member_escalation_rate]`
  - Size: `dim_member_enriched[total_lifetime_interactions]`
  - Color: `dim_member_enriched[member_segment]`
- **Visual 3 (Persona Profile Matrix):**
  - Rows: `dim_member_enriched[member_segment]`
  - Values: `Average Age`, `[Average Digital Adoption Score]`, `Average Lifetime Interactions`, `[Escalation Rate %]`

---

### Page 5: NLP Insights (Unstructured Transcripts & Friction Keywords)
- **Target Audience:** Quality Assurance, Member Experience Program Managers.
- **Visual 1 (Treemap / Bar Chart - LDA Topic Volumes):**
  - Category: `fact_interactions_enriched[dominant_topic]` (Pharmacy, Provider Access, Appointments, Billing, Claims)
  - Values: `[Total Interactions]`
  - Tooltip: `[Average Sentiment Score]`, `[Escalation Rate %]`
- **Visual 2 (Horizontal Bar Chart - High-Friction Keyword Lift):**
  - Y-Axis: Friction Keywords (`"denied"`, `"charge"`, `"twice"`, `"wait"`, `"delay"`, `"prior auth"`)
  - X-Axis: Relative Escalation Lift Factor (e.g., +3.4x)
- **Visual 3 (Complaint Volume by Queue Matrix):**
  - Rows: `fact_interactions_enriched[department]`
  - Columns: `fact_interactions_enriched[dominant_topic]`
  - Values: `[Total Escalations]`

---

### Page 6: AI Assistant Opportunity Analysis (Care Navigation RAG & Deflection ROI)
- **Target Audience:** Chief Information Officer, CFO, Digital Transformation Committee.
- **Top Financial KPI Cards:**
  1. Card 1: `[Digital Deflectable Volume]` (3,000+ phone calls in routine topics)
  2. Card 2: `[Digital Deflectable %]` (30% of total cohort volume)
  3. Card 3: `[Net Unit Cost Savings]` ($14.30 per deflected contact = $15.50 live - $1.20 AI)
  4. Card 4: `[Projected Annual Cost Savings ($)]` ($386,100+)
  5. Card 5: `[Projected 3Year Net Savings ($)]` ($1,038,300+)
- **Visual 1 (What-If Deflection Rate Parameter Slicer):**
  - Slicer Range: 10% to 50% (Default: 30%)
- **Visual 2 (Waterfall Financial Savings Model):**
  - Categories: Baseline Live Call Spend $\to$ Deflection Savings $\to$ AI Platform Maintenance $\to$ Net Annual Savings
- **Visual 3 (Deflectable Volume by Operational Topic):**
  - X-Axis: `dominant_topic` (Pharmacy, Appointments, Provider Access)
  - Values: `[Digital Deflectable Volume]`, `[Deflected Contacts Target]`
- **Visual 4 (Sensitivity Table - Financial Payback):**
  - Table showing 20%, 30%, 40% Deflection vs Gross Savings, Net Savings, and Payback Duration (3.1 to 6.3 months).
