# Kaiser Permanente Member Experience Analytics & Intelligence Pipeline
## Executive Analytics & Engineering Summary Report

**Execution Timestamp:** 2026-10-26 21:12:02  
**Total Pipeline Execution Time:** 183.50 seconds  
**Pipeline Status:** Complete (Phases 1 through 8, Phase 10)  

---

## 1. Executive Operational Baseline (Phase 1)
- **Total Interaction Volume:** 10,000 interactions across 2,621 unique members and 250 agents.
- **Overall Escalation Rate:** 17.38% (95% CI: [16.65%, 18.14%])
- **First Contact Resolution (FCR) Rate:** 63.47%
- **Mean Resolution Duration:** 17.92 minutes (Median: 14.81 min)
- **Mean Member Satisfaction (CSAT):** 3.46 / 5.0 (95% CI: [3.43, 3.48])

---

## 2. Statistical Hypothesis Testing & Confidence Intervals (Phase 2)

| Hypothesis | Test Applied | Test Statistic | p-value | Business Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **H1: Resolution Time vs. CSAT** | Spearman Rank Correlation | $\rho = -0.4628$ | 0.00e+00 | **Confirmed.** Longer resolution duration significantly degrades member satisfaction. |
| **H2: Contacts vs. Escalation** | Welch's Two-Sample t-test | $t = 32.0523$ (Cohen's $d = 1.1665$) | 2.48e-182 | **Confirmed.** Escalated interactions require 2.879 contacts vs. 1.6 for non-escalated. |
| **H3: Digital vs. Voice Escalation** | Pearson Chi-Square & Odds Ratio | $\chi^2 = 338.0008$, $OR = 2.8186$ | 1.74e-75 | **Confirmed.** Voice calls have 2.82x higher odds of escalation compared to digital channels (Phone: 25.22% vs. Digital: 10.69%). |

---

## 3. Operational Feature Engineering (Phase 4)
- Engineered 5 core features without data leakage:
  1. `Contact_Frequency_Score`: Normalized member contact volume.
  2. `Digital_Adoption_Score`: Member percentage of Chat / Mobile App interactions.
  3. `Repeat_Contact_Rate`: Proportion of member interactions requiring repeat contacts.
  4. `Service_Friction_Index`: Composite metric combining normalized resolution duration, contacts, and non-FCR penalty:
     $$\text{Friction Index} = 0.35 \times \text{NormResolution} + 0.35 \times \left(\frac{\text{Contacts}}{\max(\text{Contacts})}\right) + 0.30 \times (1 - \text{FCR})$$
  5. `Resolution_Complexity_Index`: Interaction complexity scaling department difficulty tier by duration.

---

## 4. Predictive Machine Learning: Escalation Prediction (Phase 5)
- **Leakage Safeguard:** Excluded all post-interaction variables (`Satisfaction_Score`, `Resolution_Time`, `First_Contact_Resolution`, `Service_Friction_Index`).
- **Validation:** Stratified 80/20 train/test split with 5-Fold Stratified Cross Validation.
- **Champion Algorithm:** `Logistic_Regression`
  - **Test Accuracy:** 0.764
  - **5-Fold CV ROC-AUC:** 0.784 ($\pm 0.0155$)
  - **Test ROC-AUC:** 0.7947
  - **Test PR-AUC:** 0.528
  - **Test Recall (Escalated Class):** 0.6609
  - **Test F1-Score:** 0.4936
  - **Brier Calibration Score:** 0.1778

| Evaluated Algorithm | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test Recall (Esc) | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.764 | 0.7947 | 0.528 | 0.6609 | 0.1778 |
| **Decision Tree** | 0.707 | 0.7508 | 0.4536 | 0.6782 | 0.1891 |
| **Random Forest** | 0.763 | 0.7871 | 0.4761 | 0.6523 | 0.1686 |
| **Gradient Boosting** | 0.763 | 0.7762 | 0.4776 | 0.6437 | 0.1638 |

- **Exported Artifact:** `models/escalation_model.pkl`

---

## 5. Member Behavioral Segmentation (Phase 6)
- **Clustering Methods:** K-Means ($k=4$) with `StandardScaler` & Hierarchical Agglomerative Clustering (Ward Linkage).
- **Hierarchical Clustering Metrics:** Cophenetic Correlation = 0.3566 | Adjusted Rand Index vs K-Means = 0.2408
- **Dimensionality Reduction:** PCA (2 components) explaining 50.2% of total variance.
- **Identified Behavioral Personas:**
  - **Cluster 0 — Escalation-Prone / High-Friction:** Mean Age: 50.3 yrs | Digital Adoption: 41.6% | Avg Friction: 0.481 | Escalation Rate: 35.8%
  - **Cluster 1 — Digital-First Self-Service:** Mean Age: 40.1 yrs | Digital Adoption: 81.4% | Avg Friction: 0.239 | Escalation Rate: 6.6%
  - **Cluster 2 — Service-Intensive / High-Touch:** Mean Age: 48.0 yrs | Digital Adoption: 49.5% | Avg Friction: 0.325 | Escalation Rate: 17.2%
  - **Cluster 3 — Routine Low-Need Care Seekers:** Mean Age: 57.1 yrs | Digital Adoption: 18.4% | Avg Friction: 0.212 | Escalation Rate: 9.5%

- **Exported Artifacts:** `models/kmeans_segmentation.pkl`, `reports/figures/segmentation_clusters_pca.png`, `reports/figures/hierarchical_dendrogram.png`

---

## 6. Deep Learning vs. Traditional Sentiment Benchmark (Phase 7)
- **Deep Learning Model:** PyTorch 1D-CNN with learned word embeddings (64-dim), batch normalization, adaptive pooling, and dropout.
- **Baseline Model:** TF-IDF (1-2 grams) + Logistic Regression.
- **Benchmark Evaluation (Trained on Realistic Noisy Text):**
  - **PyTorch 1D-CNN:** Accuracy: 0.882, F1-Score: 0.8979, ROC-AUC: 0.8902
  - **Baseline TF-IDF:** Accuracy: 0.882, F1-Score: 0.8979, ROC-AUC: 0.8907
- **Exported Artifact:** `models/pytorch_sentiment_model.pt`

---

## 7. NLP Topic Modeling & Complaint Classification (Phase 8)
- **Topic Modeling Algorithm:** Latent Dirichlet Allocation (LDA) with 5 Kaiser operational themes:
  - **Pharmacy:** `claim, authorization, will, clinical, doctor, review`
  - **Claims:** `check, claim, appointment, emergency, hernandez, open`
  - **Billing:** `bill, copay, ledger, look, visit, clinic`
  - **Appointments:** `appointment, physical, open, can, check, payment`
  - **Provider Access:** `care, appointment, triage, connect, pediatric, health`

- **Supervised Complaint Classification:**
  - **Overall Accuracy:** 1.0
  - **Macro F1-Score:** 1.0
  - **Weighted F1-Score:** 1.0
- **Exported Artifacts:** `models/lda_topic_model.pkl`, `models/complaint_classifier.pkl`, `reports/figures/complaint_classification_confusion_matrix.png`

---

## 8. Power BI Star Schema & RAG Staging Export Layer (Phase 10)
All 5 analytics-ready tables have been validated and exported to `data/processed/` and the root workspace:
1. `Dim_Member_Enriched.csv`: Enriched with `Member_Segment` and `Digital_Adoption_Score`.
2. `Dim_Agent.csv`: Enriched with department, shift, handled volumes, and performance ratings.
3. `Dim_Date.csv`: Continuous calendar table covering 365 calendar days with hierarchical date drill-downs.
4. `Fact_Interactions_Enriched.csv`: Transactional fact table with `Service_Friction_Index`, `Resolution_Complexity_Index`, `Predicted_Escalation_Prob`, `Dominant_Topic`, `Sentiment_Score`.
5. `RAG_Corpus_Staging.csv`: Embeddings-ready transcript table tagged with operational metadata for vector database ingestion.

---
*Report auto-generated by the Kaiser Permanente Analytics Pipeline Engine.*
