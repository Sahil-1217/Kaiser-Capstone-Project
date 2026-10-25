# Kaiser Permanente Member Experience Intelligence Platform
## Complete Office Laptop Setup, Deployment & Team Contribution Guide

> **Document Type:** Production Setup & Engineering Runbook  
> **Target Audience:** Team Engineers, Data Scientists, Analytics Developers, Office Laptops  
> **System Compatibility:** Windows 10/11, macOS Sonoma/Sequoia, Ubuntu 22.04+ LTS  
> **Version:** 2.0.0 Production Release  

---

## ⚡ 5-Minute Quick-Start Cheat Sheet

If you already have Python 3.10+ installed on your office laptop, open PowerShell or Terminal in the project root folder and execute:

```powershell
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install all required production dependencies
pip install -r requirements.txt

# 3. Create .env configuration file (copy template)
Copy-Item .env.example .env

# 4. Run the full data science pipeline + auto-sync to Neon Cloud Database
python run_pipeline.py

# 5. Launch the Executive Web Portal & Care Navigation Assistant
python -m streamlit run app.py
```
*Your application will automatically open in your browser at: `http://localhost:8501`*

---

## Table of Contents
1. [System Prerequisites & Installation](#1-system-prerequisites--installation)
2. [Project Setup & Directory Structure](#2-project-setup--directory-structure)
3. [Python Virtual Environment Configuration](#3-python-virtual-environment-configuration)
4. [Production Dependencies Installation](#4-production-dependencies-installation)
5. [Environment Variables & Database Configuration (.env)](#5-environment-variables--database-configuration-env)
6. [End-to-End Execution Guide](#6-end-to-end-execution-guide)
   - [A. Automated ML Pipeline & Cloud Database Sync](#a-automated-ml-pipeline--cloud-database-sync)
   - [B. Interactive Executive Web Application (Streamlit)](#b-interactive-executive-web-application-streamlit)
   - [C. CLI Testing of Care Navigation RAG Engine](#c-cli-testing-of-care-navigation-rag-engine)
7. [Power BI Desktop Integration (Office Laptop)](#7-power-bi-desktop-integration-office-laptop)
8. [Office Corporate Network & Firewall Troubleshooting](#8-office-corporate-network--firewall-troubleshooting)
9. [Team Contribution & Coding Standards](#9-team-contribution--coding-standards)

---

## 1. System Prerequisites & Installation

Before setting up the project on your office laptop, ensure the following core software is installed:

### A. Python (Version 3.10 to 3.13)
- **Check if Python is installed:**
  ```powershell
  python --version
  ```
- **If not installed:** Download Python from [python.org](https://www.python.org/downloads/).
- ⚠️ **CRITICAL WINDOWS STEP:** During installation, make sure to check the box:  
  **☑ Add python.exe to PATH** (at the bottom of the installer window).

### B. Git (Version Control)
- **Check if Git is installed:**
  ```powershell
  git --version
  ```
- **If not installed:** Download from [git-scm.com](https://git-scm.com/).

### C. Recommended Code Editors
- **Visual Studio Code** or **Google Antigravity IDE**
- Recommended VS Code Extensions:
  - `Python` (Microsoft)
  - `Pylance` (Microsoft)
  - `Power BI Desktop` (Microsoft, optional for viewing `.pbix` and DAX)

### D. Power BI Desktop (For Business Intelligence Deliverables)
- Install from the Microsoft Store or [Power BI Download Portal](https://powerbi.microsoft.com/desktop/).

---

## 2. Project Setup & Directory Structure

Ensure your project directory contains the following verified folder and file layout:

```text
Capstone/
├── app.py                             # 5-Tab Executive Web Portal & Care Navigation Cockpit
├── run_pipeline.py                    # Master End-to-End Pipeline Runner (Phases 1-11)
├── upload_to_cloud_db.py              # Neon Cloud PostgreSQL persistence & indexing engine
├── requirements.txt                   # Production dependency pins (Data Science, ML, RAG, DB)
├── .env.example                       # Environment variables configuration template
├── .env                               # Active cloud database credentials (create from .env.example)
├── kaiser_logo.png                    # High-resolution corporate logo for Streamlit UI
├── Dim_Member_Enriched.csv            # Star Schema: Member behavioral dimension (2,621 rows)
├── Dim_Agent.csv                      # Star Schema: Contact center agent dimension (250 rows)
├── Dim_Date.csv                       # Star Schema: Calendar dimension 2024 (365 rows)
├── Fact_Interactions_Enriched.csv     # Star Schema: Transaction fact table (10,000 rows)
├── RAG_Corpus_Staging.csv             # Staging knowledge corpus (10,002 rows)
├── data/
│   ├── raw/                           # Raw input CSV files (Member_Profile, Interaction_History, etc.)
│   ├── processed/                     # Enriched CSVs & SQLite database (kaiser_analytics.db)
│   └── evaluation_results/            # RAG audit logs (failure_log.json, benchmark_report.csv)
├── docs/
│   ├── PROJECT_MASTER_DOCUMENTATION.md # Comprehensive 10-phase technical master documentation
│   ├── rag_architecture.md            # Care Navigation RAG architectural specification
│   └── executive_deck_and_roi.md      # 15-slide executive storytelling deck & $405K ROI model
├── powerbi/
│   ├── dax_measures.dax               # 25+ production DAX measures (lower-cased schema)
│   ├── powerbi_setup_guide.md         # Step-by-step setup guide for Pages 1-6
│   ├── rls_security_rules.md          # Row-Level Security definitions
│   └── star_schema_relationships.md   # Cardinality matrix (1:* single direction)
├── reports/
│   ├── figures/                       # Publication-grade figures (ROC/PR, PCA, LDA, Distributions)
│   └── hypothesis_testing_results.json# Welch's t-test, Spearman correlation, Wilson CIs
├── src/
│   ├── config.py                      # Global path configuration & file schema mappings
│   ├── db_sync.py                     # SQLAlchemy persistence manager (SQLite & PostgreSQL)
│   ├── rag_engine.py                  # Care Navigation RAG Engine (Hybrid RRF, NER, NLI Verifier)
│   ├── module1_data_integration.py    # Phase 1: Ingestion & referential integrity validation
│   ├── module2_hypotheses.py          # Phase 2: Statistical hypothesis testing
│   ├── module3_eda.py                 # Phase 3: Automated visual analytics
│   ├── module4_feature_engineering.py # Phase 4: Zero-leakage features (Friction Index, Digital Score)
│   ├── module5_ml_modeling.py         # Phase 5: Balanced Logistic Regression escalation classifier
│   ├── module6_segmentation.py        # Phase 6: K-Means (k=4) behavioral persona clustering
│   ├── module7_sentiment_dl.py        # Phase 7: PyTorch 1D-CNN sentiment score classifier
│   ├── module8_nlp_topics.py          # Phase 8: Latent Dirichlet Allocation (5 operational themes)
│   └── module9_star_schema.py         # Phase 10: Star schema assembly & dimension generator
└── MEDICAL_RAG_KP/                    # Reference Agentic Medical RAG repository (WHO/CDC guidelines)
```

---

## 3. Python Virtual Environment Configuration

Always run this project in an isolated virtual environment to prevent dependency conflicts with your office laptop's system packages.

### Windows (PowerShell / Command Prompt)
1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd C:\path\to\Capstone
   ```
2. Create the virtual environment named `venv`:
   ```powershell
   python -m venv venv
   ```
3. Activate the virtual environment:
   - **PowerShell:**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
     > **Troubleshooting PowerShell Script Execution Error:**  
     > If you see `execution of scripts is disabled on this system`, run this command once:  
     > ```powershell
     > Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
     > .\venv\Scripts\Activate.ps1
     > ```
   - **Command Prompt (cmd.exe):**
     ```cmd
     venv\Scripts\activate.bat
     ```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

*When successfully activated, `(venv)` will appear at the start of your terminal prompt.*

---

## 4. Production Dependencies Installation

With `(venv)` activated, upgrade `pip` and install all required packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Verification Script
Verify that your environment is 100% operational by executing:
```powershell
python -c "
import streamlit, torch, sqlalchemy, psycopg2, rank_bm25, sklearn, pandas, numpy
print('==============================================')
print('>>> ALL DEPENDENCIES VERIFIED SUCCESSFULLY <<<')
print('==============================================')
"
```

If it prints `>>> ALL DEPENDENCIES VERIFIED SUCCESSFULLY <<<`, your office laptop is ready.

---

## 5. Environment Variables & Database Configuration (.env)

The platform supports **Dual Database Persistence**:
1. **Local SQLite (`data/kaiser_analytics.db`):** Requires zero credentials or internet connection.
2. **Neon Cloud PostgreSQL:** Remote, serverless cloud database for multi-user access and Power BI DirectQuery.

### Setting Up `.env`
1. Copy the `.env.example` file to `.env`:
   - **Windows PowerShell:**
     ```powershell
     Copy-Item .env.example .env
     ```
   - **macOS / Linux:**
     ```bash
     cp .env.example .env
     ```

2. Open `.env` in any text editor and populate the active **Neon Cloud PostgreSQL credentials**:
   ```dotenv
   # Neon Cloud PostgreSQL Connection String (Production Branch)
   DATABASE_URL="postgresql+psycopg2://neondb_owner:npg_YrA5gGhDfpn8@ep-rough-wave-b4tlujf7-pooler.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require"
   DATABASE_URL_UNPOOLED="postgresql+psycopg2://neondb_owner:npg_YrA5gGhDfpn8@ep-rough-wave-b4tlujf7.c-6.us-east-2.aws.neon.tech/neondb?sslmode=require"
   NEON_BRANCH=production
   ```

> [!NOTE]
> **Office Firewall Note:** If your office corporate proxy restricts outbound port `5432` to Neon Cloud, the system automatically falls back to the local SQLite database (`data/kaiser_analytics.db`), so you can continue working without interruption.

---

## 6. End-to-End Execution Guide

### A. Automated ML Pipeline & Cloud Database Sync
To execute the complete 11-step data engineering, statistical testing, machine learning, and cloud synchronization workflow, run:

```powershell
python run_pipeline.py
```

#### What happens under the hood:
1. **Step 1:** Ingests the 5 raw CSVs and validates 100% referential integrity.
2. **Step 2:** Executes statistical hypothesis testing (Welch's t-test, Spearman correlation, Wilson Score CIs).
3. **Step 3:** Generates 4 publication-grade figures in `reports/figures/`.
4. **Step 4:** Builds zero-leakage pre-interaction features (`Service_Friction_Index`, `Digital_Adoption_Score`).
5. **Step 5:** Trains the champion Balanced Logistic Regression escalation model (Test ROC-AUC: 0.795, Recall: 66.1%).
6. **Step 6:** Trains K-Means ($k=4$) member behavioral persona clustering and saves 2D PCA visual.
7. **Step 7:** Trains PyTorch 1D-CNN continuous sentiment classifier.
8. **Step 8:** Executes Latent Dirichlet Allocation (LDA) to extract the 5 Kaiser Permanente operational complaint themes.
9. **Step 9:** Assembles the Star Schema (4 tables + RAG corpus staging).
10. **Step 10:** Synchronizes Star Schema tables to local SQLite (`data/kaiser_analytics.db`).
11. **Step 11:** Connects to **Neon Cloud PostgreSQL** with SSL, lowercases all table and column names for standard SQL queries, uploads data in chunks, and generates primary key/foreign key indexes.

*Total runtime: ~2 minutes. Exits with code 0.*

---

### B. Interactive Executive Web Application (Streamlit)
To launch the full interactive web application, run:

```powershell
python -m streamlit run app.py
```

*Always use `python -m streamlit run app.py` rather than `streamlit run app.py` to ensure the command uses the active virtual environment.*

#### Application Navigation:
- **Top Left Logo:** Official Kaiser Permanente corporate brand logo.
- **Sidebar Control Center:**
  - Raw batch CSV upload.
  - Relational Database Backend selector (Neon Cloud PostgreSQL vs Local SQLite).
  - Pipeline Execution trigger button with embedded real-time log terminal.
  - **RAG Engine Parameters:** Slider controls for Top-K candidates (3–8), Hybrid Balance (BM25 vs Vector), NLI Fact-Checker toggle, and Biomedical NER toggle.
- **Tab 1: Executive KPI Overview:** Operational metrics, 95% Wilson confidence intervals, and hypothesis confirmation badges.
- **Tab 2: Machine Learning & Personas:** Interactive ROC/PR curves, K-Means persona PCA clusters, and LDA topic distributions.
- **Tab 3: Live Star Schema Inspector:** Real-time SQL query interface against Neon Cloud or SQLite with row limit sliders, escalation filters, and one-click CSV export.
- **Tab 4: Power BI & RAG Connection Guides:** Direct download buttons for DAX files, RLS rules, and executive storytelling decks.
- **Tab 5: AI Care Navigation Assistant (MediRAG Powered 4-Sub-Tab Cockpit):**
  - **Sub-Tab 5.1 (💬 Clinical & Care Navigation Console):** 7 interactive preset buttons (90-Day Refill, Pediatric Booking, Claims Dispute, Specialist Prior Auth, Emergency Check, WHO Metformin, CDC Pneumonia), live step callbacks with toasts, telemetry badges, grounded response card, biomedical entity pills, citations, CRM screen-pop with simulate push button, claim-by-claim NLI verification table, and HIPAA redaction telemetry.
  - **Sub-Tab 5.2 (📚 Dynamic Knowledge Corpus Manager):** Corpus browser with department and keyword filters, plus a live form to dynamically ingest new clinical guidelines into the active retriever.
  - **Sub-Tab 5.3 (🛡️ Failure Mode & Hallucination Audit Trace):** Auditable tracking of retrieval drops, contradictions, and emergency intercepts with summary KPI cards and CSV export.
  - **Sub-Tab 5.4 (📊 Systematic Benchmark Matrix & RAGAS Analytics):** Experimental performance comparisons (Lexical vs Dense vs Hybrid RRF vs Re-Ranked Hybrid) with interactive bar charts for Faithfulness, Relevancy, and Latency.

---

### C. CLI Testing of Care Navigation RAG Engine
You can test the Care Navigation Assistant directly from the terminal without opening a browser:

- **Run an individual inquiry with full telemetry:**
  ```powershell
  python src/rag_engine.py --query "What is the starting dose of Metformin for Type 2 Diabetes?"
  ```
- **Execute all 7 automated benchmark dialogues:**
  ```powershell
  python src/rag_engine.py
  ```

---

## 7. Power BI Desktop Integration (Office Laptop)

To build or refresh the official 6-page Power BI report on your office laptop:

### Step 1: Connect to Neon Cloud Database
1. Open **Power BI Desktop**.
2. Click **Get Data** $\to$ **PostgreSQL Database** $\to$ **Connect**.
3. In the connection window, enter:
   - **Server:** `ep-rough-wave-b4tlujf7-pooler.c-6.us-east-2.aws.neon.tech`
   - **Database:** `neondb`
   - **Data Connectivity Mode:** `Import` (or `DirectQuery`)
4. When prompted for credentials, select the **Database** tab (on the left menu) and enter:
   - **User name:** `neondb_owner`
   - **Password:** `npg_YrA5gGhDfpn8`
5. Check the box for the 4 Star Schema tables:
   - `fact_interactions_enriched`
   - `dim_member_enriched`
   - `dim_agent`
   - `dim_date`
6. Click **Load**.

### Step 2: Establish Star Schema Relationships (Model View)
In Power BI **Model View**, verify the single-direction 1-to-many relationships:
- `dim_member_enriched[member_id]` (1) $\to$ `fact_interactions_enriched[member_id]` (*)
- `dim_agent[agent_id]` (1) $\to$ `fact_interactions_enriched[agent_id]` (*)
- `dim_date[date_key]` (1) $\to$ `fact_interactions_enriched[date_key]` (*)

### Step 3: Import DAX Measures
Open [`powerbi/dax_measures.dax`](file:///c:/Users/sahil/OneDrive/Desktop/Capstone/powerbi/dax_measures.dax). Copy and paste the pre-calibrated measures (e.g. `[Total Interactions]`, `[Escalation Rate %]`, `[First Contact Resolution Rate %]`, `[Projected Annual Cost Savings ($)]`) directly into your report.

---

## 8. Office Corporate Network & Firewall Troubleshooting

### Issue 1: `pip install` fails with SSL Certificate Verify Error
- **Cause:** Office proxy or Zscaler / corporate firewall intercepts SSL traffic.
- **Solution:** Pass trusted hosts to pip:
  ```powershell
  pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
  ```

### Issue 2: `streamlit : The term 'streamlit' is not recognized`
- **Cause:** Python's `Scripts/` folder is not in your system environment PATH.
- **Solution:** Always run via Python module syntax:
  ```powershell
  python -m streamlit run app.py
  ```

### Issue 3: `Set-ExecutionPolicy` Error in PowerShell
- **Cause:** Windows restricts script execution for non-administrator accounts.
- **Solution:** Apply execution policy to the current process only (does not require Administrator privileges):
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
  .\venv\Scripts\Activate.ps1
  ```

### Issue 4: Port 8501 is already in use
- **Cause:** Another Streamlit process is running in the background.
- **Solution:** Launch on an alternate port:
  ```powershell
  python -m streamlit run app.py --server.port 8502
  ```

### Issue 5: Port 5432 Blocked by Office Network
- **Cause:** Corporate security blocks outbound traffic on PostgreSQL port `5432`.
- **Solution:** The platform works 100% locally with SQLite (`data/kaiser_analytics.db`). In the Streamlit sidebar, select **"SQLite (Local Zero-Setup)"** from the dropdown.

---

## 9. Team Contribution & Coding Standards

When contributing code, models, or documentation to the repository:

### A. Git Branching Strategy
```bash
# Create a new feature branch
git checkout -b feature/care-navigation-enhancement

# Make your changes and verify compilation
python -c "import py_compile; py_compile.compile('app.py', doraise=True)"
python -c "import py_compile; py_compile.compile('src/rag_engine.py', doraise=True)"

# Commit with a clean message
git commit -m "feat(rag): add pediatric telehealth guideline to hybrid search space"

# Push to remote repository
git push origin feature/care-navigation-enhancement
```

### B. Python Code Guidelines
- Follow **PEP 8** style guidelines with type annotations (`from typing import List, Dict, Optional`).
- Use raw string literals (`r"""..."""`) for regex strings to avoid `SyntaxWarning: invalid escape sequence`.
- Keep database table and column names strictly lowercase (`fact_interactions_enriched`, `date_key`, `member_id`) to ensure seamless PostgreSQL and Power BI compatibility.
- Ensure all automated unit benchmarks pass before submitting pull requests:
  ```powershell
  python src/rag_engine.py
  ```

---

*For technical architecture details, statistical formulas, and mathematical derivations, refer to the [Project Master Documentation](file:///c:/Users/sahil/OneDrive/Desktop/Capstone/docs/PROJECT_MASTER_DOCUMENTATION.md).*
