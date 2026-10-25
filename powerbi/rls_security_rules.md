# Kaiser Permanente Member Experience Intelligence Platform
## Production Row-Level Security (RLS) Configuration Guide (Phase 10)

Row-Level Security (RLS) restricts data access for given users based on their operational role, department, or geographic region, ensuring HIPAA data governance and strict least-privilege compliance.

---

## 1. Security Roles Architecture

We define 4 distinct enterprise security roles within Power BI Desktop:

| Role Name | Target User | Filter Scope | Primary Security Objective |
| :--- | :--- | :--- | :--- |
| **`Department_Manager`** | Department Operations Heads (Billing, Pharmacy, Member Services, Clinical Triage) | Filters `Fact_Interactions_Enriched` and `Dim_Agent` to the manager's assigned department. | Restricts cross-departmental operational visibility. |
| **`Regional_Director`** | Regional Care Experience Directors (Northern CA, Southern CA, Colorado, etc.) | Filters `Dim_Member_Enriched` by member `Region`. Cascades down to `Fact_Interactions_Enriched`. | Ensures regional operational autonomy. |
| **`Queue_Supervisor`** | Contact Center Team Leads | Filters `Fact_Interactions_Enriched[Queue]` to assigned queue queues. | Supervisors only view their direct queue operations. |
| **`Executive_Read_All`** | CCO, Chief Medical Officer, VP of Health Plan Operations | No filter expressions applied. | Full enterprise visibility across all regions and departments. |

---

## 2. DAX Filter Expressions for Power BI Desktop

To configure RLS in Power BI Desktop:
1. Navigate to the **Modeling** ribbon.
2. Click **Manage Roles** $\to$ **Create**.
3. Apply the DAX filter expressions detailed below.

### Role 1: `Department_Manager` (Dynamic User Principal Name)
*Filters interactions and agent dimensions to the user's logged-in department.*

**On table `Fact_Interactions_Enriched`:**
```dax
[Department] = 
LOOKUPVALUE(
    User_Department_Mapping[Department],
    User_Department_Mapping[UserPrincipalName],
    USERPRINCIPALNAME()
)
```
*Static fallback for testing:*
```dax
[Department] = "Billing & Claims"
```

---

### Role 2: `Regional_Director` (Geographic Partition)
*Filters the member dimension to their regional jurisdiction. Because of the 1:* relationship, this automatically filters the fact table.*

**On table `Dim_Member_Enriched`:**
```dax
[Region] = "Northern California"
```
*Dynamic regional rule:*
```dax
[Region] = 
LOOKUPVALUE(
    User_Region_Mapping[Region],
    User_Region_Mapping[UserPrincipalName],
    USERPRINCIPALNAME()
)
```

---

### Role 3: `Queue_Supervisor`
*Restricts supervisors to Tier 2 Grievance or Claims queues.*

**On table `Fact_Interactions_Enriched`:**
```dax
[Queue] IN {
    "Tier 2 Escalations & Grievance",
    "Claims Adjudication Queue"
}
```

---

### Role 4: `Executive_Read_All`
*Leave all table filters blank.* Allows unrestricted access to aggregate KPIs, ML probabilities, and Star Schema tables.

---

## 3. Validation & Testing Procedure ("View as Role")

Before publishing the `.pbix` report to the Power BI Service:
1. In the **Modeling** tab, click **View as Roles**.
2. Check `Regional_Director` (or input a test UPN e.g., `director.norcal@kp.org`).
3. **Verify:**
   - On **Page 1 (Executive KPI)**: Total Interactions drops from `10,000` to the specific regional cohort (e.g. `~3,500` for Northern California).
   - On **Page 3 (Personas)**: Member segments reflect only Northern California members.
   - Slicers for `Region` become locked to `"Northern California"`.
4. Click **Stop Viewing**.
5. Repeat for `Department_Manager` and verify that `Department` slicers and visuals show only the selected department.

---

## 4. Power BI Service Cloud Deployment
1. Publish report to the target workspace (e.g., `KP-Member-Experience-Prod`).
2. In Power BI Service, locate the published dataset $\to$ click the ellipsis `...` $\to$ **Security**.
3. Add Azure Active Directory (AAD) / Entra ID Security Groups to each role:
   - `Department_Manager` $\to$ `SG_KP_ContactCenter_Managers`
   - `Regional_Director` $\to$ `SG_KP_Regional_Leadership`
   - `Executive_Read_All` $\to$ `SG_KP_Executive_Leadership`
4. Save and test in service using "Test as role".
