# Kaiser Permanente Capstone 3: Member Experience, Contact Center & Care Navigation Intelligence

A production-grade, modular, and reproducible Python analytics and intelligence pipeline built to analyze member interaction friction, predict service escalations, extract unstructured insights from contact transcripts, cluster members into actionable behavioral segments, and export an analytics-ready Star Schema for executive Power BI dashboards.

---

## 1. Project Architecture

```
Capstone/
│
├── data/
│   ├── raw/                           # Ingested raw source CSVs
│   │   ├── Member_Profile.csv
│   │   ├── Interaction_History.csv
│   │   ├── Service_Performance.csv
│   │   ├── Operations_Data.csv
│   │   └── Transcript_Data.csv
│   └── processed/                     # Power BI Star Schema & RAG Exports
│       ├── Dim_Member_Enriched.csv
│       ├── Dim_Agent.csv
│       ├── Dim_Date.csv
│       ├── Fact_Interactions_Enriched.csv
│       └── RAG_Corpus_Staging.csv
│
├── models/                            # Serialized Machine Learning & DL Artifacts
│   ├── escalation_model.pkl           # Champion Escalation Predictor (Pipeline)
│   ├── kmeans_segmentation.pkl        # KMeans & Scaler & Persona Artifacts
│   ├── pytorch_sentiment_model.pt     # PyTorch 1D-CNN Sentiment Classifier Weights
│   └── lda_topic_model.pkl            # Trained LDA Topic Model & Vectorizer
│
├── reports/                           # Executive Reports & Tabular Summaries
│   ├── figures/                       # Publication-Grade Visualizations (300 DPI)
│   │   ├── eda_distributions.png
│   │   ├── eda_correlations.png
│   │   ├── eda_channel_escalation.png
│   │   ├── eda_reason_resolution.png
│   │   ├── ml_roc_pr_curves.png
│   │   ├── segmentation_clusters_pca.png
│   │   ├── sentiment_benchmark.png
│   │   └── topic_distributions.png
│   ├── hypothesis_testing_results.json
│   ├── ml_model_evaluation.json
│   ├── segmentation_report.json
│   ├── sentiment_benchmark_results.json
│   ├── nlp_topic_report.json
│   └── pipeline_summary_report.md
│
├── src/                               # Modular Python Package
│   ├── __init__.py
│   ├── config.py                      # Global Paths, Constants, Weights & Seeds
│   ├── data_generator.py              # Calibrated Kaiser Permanente Synthetic Generator
│   ├── module1_data_integration.py    # Schema Integrity & Referential Key Checks
│   ├── module2_hypothesis_testing.py  # Spearman, Welch t-test, Chi-Square & 95% CIs
│   ├── module3_eda.py                 # Automated Distribution & Correlation Plots
│   ├── module4_feature_engineering.py # Leakage-Free Composite Indices & Metrics
│   ├── module5_escalation_ml.py       # Pre-Interaction ML Benchmarking & Prediction
│   ├── module6_segmentation.py        # Member Aggregation, KMeans (k=4), PCA & Personas
│   ├── module7_sentiment_dl.py        # PyTorch 1D-CNN vs Baseline TF-IDF Benchmark
│   ├── module8_nlp_topics.py          # Lemmatization, Friction Terms & 5-Theme LDA
│   └── module9_star_schema.py         # Star Schema & RAG Corpus Export Layer
│
├── run_pipeline.py                    # Master CLI Pipeline Orchestrator
├── requirements.txt                   # Environment Dependencies
└── README.md
```

---

## 2. Quickstart & Execution

### Installation
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline End-to-End
Execute all phases sequentially with unified logging:
```bash
python run_pipeline.py
```

### CLI Arguments
- `--generate-data`: Regenerate the statistically calibrated dataset.
- `--skip-eda`: Fast execution bypassing visualization image generation.

Each module can also be tested and executed independently:
```bash
python src/module1_data_integration.py
python src/module2_hypothesis_testing.py
python src/module3_eda.py
python src/module4_feature_engineering.py
python src/module5_escalation_ml.py
python src/module6_segmentation.py
python src/module7_sentiment_dl.py
python src/module8_nlp_topics.py
python src/module9_star_schema.py
```

---

## 3. Module & Phase Highlights

### Module 1: Data Integration & Schema Integrity (Phase 1)
- Validates 100% foreign key referential integrity between `Member_ID` (Profile $\leftrightarrow$ History) and `Interaction_ID` across all 4 interaction tables.
- Joins data sources into a denormalized staging dataframe.
- Calculates baseline KPIs: Escalation Rate (17.38%), FCR Rate (63.47%), Mean Handle Duration (17.92 min), Mean CSAT (3.46 / 5.0).

### Module 2: Statistical Hypothesis Testing (Phase 2)
1. **Hypothesis 1 (Resolution Time vs. CSAT):** Spearman Rank Correlation $\rho = -0.4628$ ($p < 10^{-15}$). Every 10-minute handle time increase significantly reduces satisfaction.
2. **Hypothesis 2 (Contacts vs. Escalation):** Welch's two-sample t-test $t = 32.05$, $p = 2.48 \times 10^{-182}$, Cohen's $d = 1.167$. Escalated interactions average 2.88 contacts vs. 1.60 for non-escalated.
3. **Hypothesis 3 (Digital vs. Voice Escalations):** Pearson Chi-Square $\chi^2 = 338.00$ ($p < 10^{-74}$), Odds Ratio = 2.82x (95% CI: [2.52, 3.16]). Voice interactions are 2.8x more likely to escalate than digital chat/app.
4. **95% Confidence Intervals:** Wilson Score Escalation Rate: [16.65%, 18.14%]; Mean CSAT: [3.43, 3.48].

### Module 3: Exploratory Data Analysis (EDA) Automation (Phase 3)
- Multi-panel distributions of duration, CSAT, and contact frequency.
- Heatmap correlation matrix of numerical metrics.
- Bivariate breakdowns for Channel vs Escalation/FCR and Reason vs Duration.

### Module 4: Operational Feature Engineering (Phase 4)
- **Zero-Leakage Safeguard:** Features strictly separated into pre-interaction predictors and post-interaction diagnostics.
- Engineered Features:
  - `Contact_Frequency_Score`: Normalized interaction volume per member.
  - `Digital_Adoption_Score`: Proportion of interactions via Chat or Mobile App ($\in [0, 1]$).
  - `Repeat_Contact_Rate`: Proportion of repeat inquiry interactions.
  - `Service_Friction_Index`: $0.35 \times \text{NormResolution} + 0.35 \times \text{NormContacts} + 0.30 \times (1 - \text{FCR})$.
  - `Resolution_Complexity_Index`: Interaction complexity scaling department difficulty tier by duration.

### Module 5: Escalation Prediction Machine Learning (Phase 5)
- Class-imbalanced classification target (`Escalation_Flag`).
- Strict exclusion of post-interaction metrics (`Satisfaction_Score`, `Resolution_Time`, `FCR`, and `Service_Friction_Index`).
- Pre-interaction predictors: Age, Plan Type, Region, Channel, Reason, Department, Queue, Shift, Contact Frequency, Digital Adoption, Contacts.
- Benchmark: Logistic Regression, Random Forest, HistGradientBoosting with 5-Fold Stratified Cross Validation.
- Champion Model: Logistic Regression (balanced) achieves Test ROC-AUC of **0.7947**, PR-AUC of **0.528**, and 66.1% Recall on escalations.
- Inferred `Predicted_Escalation_Prob` appended to dataset.

### Module 6: Member Behavioral Segmentation (Phase 6)
- Member-level aggregation of friction, contacts, digital adoption, age, and complexity.
- K-Means ($k=4$) validated with Silhouette (0.161) and Davies-Bouldin (1.646).
- 2D PCA projection with cluster centroids.
- Business Personas:
  - **Cluster 0:** *Escalation-Prone / High-Friction* (27.5% of members; 35.8% escalation rate, 0.481 avg friction).
  - **Cluster 1:** *Digital-First Self-Service* (28.2% of members; 81.4% digital adoption, 6.6% escalation rate).
  - **Cluster 2:** *Service-Intensive / High-Touch* (20.1% of members; older, moderate friction, repeat phone contacts).
  - **Cluster 3:** *Routine Low-Need Care Seekers* (24.2% of members; low friction, low contact frequency).

### Module 7: Deep Learning Sentiment Benchmark (Phase 7)
- PyTorch 1D-CNN with learned 64-dim word embeddings, 1D convolution, adaptive max pooling, batch normalization, and dropout.
- Benchmarked against TF-IDF (1-2 grams) + Logistic Regression.
- Loss curves and performance metrics exported to `reports/sentiment_benchmark_results.json`.
- Continuous `Sentiment_Score` appended to dataset.

### Module 8: NLP Topic Modeling & Complaint Classification (Phase 8)
- Transcript cleaning, stopword removal, and rule-based lemmatization.
- TF-IDF high-friction term contrast: identifies terms like "charge", "twice", "wait", "delay", "denied" with highest escalation lift.
- Latent Dirichlet Allocation (LDA) classifies transcripts into 5 core Kaiser operational themes:
  - *Appointments*
  - *Billing*
  - *Claims*
  - *Pharmacy*
  - *Provider Access*
- Appends `Dominant_Topic` and confidence scores.

### Module 9: Power BI Star Schema & RAG Interface (Phase 10)
Structured tables ready for immediate drag-and-drop into Power BI:
1. `Dim_Member_Enriched.csv`: `Member_ID`, `Age`, `Plan_Type`, `Region`, `Member_Segment`, `Digital_Adoption_Score`, `Total_Lifetime_Interactions`, `Avg_Member_Friction`.
2. `Dim_Agent.csv`: `Agent_ID`, `Queue`, `Department`, `Shift`, `Total_Interactions_Handled`, `Agent_FCR_Rate`, `Agent_Avg_CSAT`.
3. `Dim_Date.csv`: Continuous calendar table spanning 365 days with Year, Quarter, Month, Month Name, Day, Day Name, Week Number, Is Weekend.
4. `Fact_Interactions_Enriched.csv`: Core transactional fact table with `Service_Friction_Index`, `Resolution_Complexity_Index`, `Predicted_Escalation_Prob`, `Dominant_Topic`, `Sentiment_Score`.
5. `RAG_Corpus_Staging.csv`: Embeddings-ready corpus table with clean text, operational metadata tags, and sentiment scores.

---

## 4. Power BI Modeling Instructions

1. **Open Power BI Desktop** $\rightarrow$ **Get Data** $\rightarrow$ **Text/CSV**.
2. Load all 5 tables from `data/processed/` (or root directory).
3. In Model View, configure the relationships:
   - `Dim_Member_Enriched[Member_ID]` $1 \rightarrow \infty$ `Fact_Interactions_Enriched[Member_ID]` (Single direction)
   - `Dim_Agent[Agent_ID]` $1 \rightarrow \infty$ `Fact_Interactions_Enriched[Agent_ID]` (Single direction)
   - `Dim_Date[Date]` $1 \rightarrow \infty$ `Fact_Interactions_Enriched[Interaction_Date]` (Single direction)
4. Recommended DAX KPI Measures:
   - `Escalation Rate = DIVIDE(CALCULATE(COUNTROWS(Fact_Interactions_Enriched), Fact_Interactions_Enriched[Escalation_Flag] = 1), COUNTROWS(Fact_Interactions_Enriched))`
   - `FCR Rate = AVERAGE(Fact_Interactions_Enriched[First_Contact_Resolution])`
   - `Average Friction = AVERAGE(Fact_Interactions_Enriched[Service_Friction_Index])`
   - `Mean CSAT = AVERAGE(Fact_Interactions_Enriched[Satisfaction_Score])`
   - `High Risk Escalations = CALCULATE(COUNTROWS(Fact_Interactions_Enriched), Fact_Interactions_Enriched[Predicted_Escalation_Prob] >= 0.50)`
