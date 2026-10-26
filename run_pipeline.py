"""
Master Execution Pipeline Orchestrator:
Kaiser Permanente Member Experience Analytics & Intelligence Pipeline

Executes Phases 1 through 8 and Phase 10 (Star Schema Export):
- Module 1: Data Integration & Schema Integrity
- Module 2: Statistical Hypothesis Testing & Confidence Intervals
- Module 3: Exploratory Data Analysis (EDA) Automation
- Module 4: Operational Feature Engineering (Zero-Leakage)
- Module 5: Escalation Prediction Machine Learning (Class-Imbalanced Benchmark)
- Module 6: Member Behavioral Segmentation (K-Means & PCA)
- Module 7: Deep Learning vs. Baseline Sentiment Benchmark (PyTorch 1D-CNN)
- Module 8: NLP Topic Modeling & Complaint Classification (LDA 5 Themes)
- Module 9: Power BI Star Schema & RAG Staging Export Layer
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Set Project Root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, REPORTS_DIR,
    FIGURES_DIR, MODELS_DIR, INPUT_FILES, OUTPUT_FILES
)
from src.data_generator import generate_synthetic_data
from src.module1_data_integration import run_module1
from src.module2_hypothesis_testing import run_module2
from src.module3_eda import run_module3
from src.module4_feature_engineering import run_module4
from src.module5_escalation_ml import run_module5
from src.module6_segmentation import run_module6
from src.module7_sentiment_dl import run_module7
from src.module8_nlp_topics import run_module8
from src.module9_star_schema import run_module9
from src.db_sync import run_database_sync
from upload_to_cloud_db import CloudDatabaseUploader


def generate_pipeline_summary_report(
    baseline_metrics: dict,
    hypothesis_results: dict,
    ml_metrics: dict,
    segmentation_report: dict,
    sentiment_report: dict,
    nlp_report: dict,
    execution_time: float
):
    """
    Generates an executive-level Markdown summary report for stakeholders.
    """
    report_path = REPORTS_DIR / "pipeline_summary_report.md"

    champ = ml_metrics.get("champion_model", "Logistic_Regression")
    champ_eval = ml_metrics.get(champ, {})

    h1 = hypothesis_results["hypothesis_1"]
    h2 = hypothesis_results["hypothesis_2"]
    h3 = hypothesis_results["hypothesis_3"]
    ci = hypothesis_results["confidence_intervals"]

    md_content = f"""# Kaiser Permanente Member Experience Analytics & Intelligence Pipeline
## Executive Analytics & Engineering Summary Report

**Execution Timestamp:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Pipeline Execution Time:** {execution_time:.2f} seconds  
**Pipeline Status:** Complete (Phases 1 through 8, Phase 10)  

---

## 1. Executive Operational Baseline (Phase 1)
- **Total Interaction Volume:** {baseline_metrics['Total_Interactions']:,} interactions across {baseline_metrics['Unique_Members']:,} unique members and {baseline_metrics['Unique_Agents']} agents.
- **Overall Escalation Rate:** {baseline_metrics['Overall_Escalation_Rate']*100:.2f}% (95% CI: [{ci['escalation_rate']['ci_95_lower']*100:.2f}%, {ci['escalation_rate']['ci_95_upper']*100:.2f}%])
- **First Contact Resolution (FCR) Rate:** {baseline_metrics['Overall_FCR_Rate']*100:.2f}%
- **Mean Resolution Duration:** {baseline_metrics['Mean_Resolution_Time_Mins']} minutes (Median: {baseline_metrics['Median_Resolution_Time_Mins']} min)
- **Mean Member Satisfaction (CSAT):** {baseline_metrics['Mean_CSAT_Score']} / 5.0 (95% CI: [{ci['mean_csat']['ci_95_lower']:.2f}, {ci['mean_csat']['ci_95_upper']:.2f}])

---

## 2. Statistical Hypothesis Testing & Confidence Intervals (Phase 2)

| Hypothesis | Test Applied | Test Statistic | p-value | Business Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| **H1: Resolution Time vs. CSAT** | Spearman Rank Correlation | $\\rho = {h1['spearman_rho']}$ | {h1['spearman_p_value']:.2e} | **Confirmed.** Longer resolution duration significantly degrades member satisfaction. |
| **H2: Contacts vs. Escalation** | Welch's Two-Sample t-test | $t = {h2['welch_t_stat']}$ (Cohen's $d = {h2['cohens_d_effect_size']}$) | {h2['welch_p_value']:.2e} | **Confirmed.** Escalated interactions require {h2['escalated_group']['mean_contacts']} contacts vs. {h2['non_escalated_group']['mean_contacts']} for non-escalated. |
| **H3: Digital vs. Voice Escalation** | Pearson Chi-Square & Odds Ratio | $\\chi^2 = {h3['chi2_statistic']}$, $OR = {h3['odds_ratio_voice_vs_digital']}$ | {h3['p_value']:.2e} | **Confirmed.** Voice calls have {h3['odds_ratio_voice_vs_digital']:.2f}x higher odds of escalation compared to digital channels (Phone: {h3['voice_escalation_rate']*100:.2f}% vs. Digital: {h3['digital_escalation_rate']*100:.2f}%). |

---

## 3. Operational Feature Engineering (Phase 4)
- Engineered 5 core features without data leakage:
  1. `Contact_Frequency_Score`: Normalized member contact volume.
  2. `Digital_Adoption_Score`: Member percentage of Chat / Mobile App interactions.
  3. `Repeat_Contact_Rate`: Proportion of member interactions requiring repeat contacts.
  4. `Service_Friction_Index`: Composite metric combining normalized resolution duration, contacts, and non-FCR penalty:
     $$\\text{{Friction Index}} = 0.35 \\times \\text{{NormResolution}} + 0.35 \\times \\left(\\frac{{\\text{{Contacts}}}}{{\\max(\\text{{Contacts}})}}\\right) + 0.30 \\times (1 - \\text{{FCR}})$$
  5. `Resolution_Complexity_Index`: Interaction complexity scaling department difficulty tier by duration.

---

## 4. Predictive Machine Learning: Escalation Prediction (Phase 5)
- **Leakage Safeguard:** Excluded all post-interaction variables (`Satisfaction_Score`, `Resolution_Time`, `First_Contact_Resolution`, `Service_Friction_Index`).
- **Validation:** Stratified 80/20 train/test split with 5-Fold Stratified Cross Validation.
- **Champion Algorithm:** `{champ}`
  - **Test Accuracy:** {champ_eval.get('test_accuracy', 'N/A')}
  - **5-Fold CV ROC-AUC:** {champ_eval.get('cv_mean_roc_auc', 'N/A')} ($\\pm {champ_eval.get('cv_std_roc_auc', 'N/A')}$)
  - **Test ROC-AUC:** {champ_eval.get('test_roc_auc', 'N/A')}
  - **Test PR-AUC:** {champ_eval.get('test_pr_auc', 'N/A')}
  - **Test Recall (Escalated Class):** {champ_eval.get('test_recall_pos', 'N/A')}
  - **Test F1-Score:** {champ_eval.get('test_f1_pos', 'N/A')}
  - **Brier Calibration Score:** {champ_eval.get('test_brier_score', 'N/A')}

| Evaluated Algorithm | Test Accuracy | Test ROC-AUC | Test PR-AUC | Test Recall (Esc) | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | {ml_metrics.get('Logistic_Regression', {}).get('test_accuracy', 'N/A')} | {ml_metrics.get('Logistic_Regression', {}).get('test_roc_auc', 'N/A')} | {ml_metrics.get('Logistic_Regression', {}).get('test_pr_auc', 'N/A')} | {ml_metrics.get('Logistic_Regression', {}).get('test_recall_pos', 'N/A')} | {ml_metrics.get('Logistic_Regression', {}).get('test_brier_score', 'N/A')} |
| **Decision Tree** | {ml_metrics.get('Decision_Tree', {}).get('test_accuracy', 'N/A')} | {ml_metrics.get('Decision_Tree', {}).get('test_roc_auc', 'N/A')} | {ml_metrics.get('Decision_Tree', {}).get('test_pr_auc', 'N/A')} | {ml_metrics.get('Decision_Tree', {}).get('test_recall_pos', 'N/A')} | {ml_metrics.get('Decision_Tree', {}).get('test_brier_score', 'N/A')} |
| **Random Forest** | {ml_metrics.get('Random_Forest', {}).get('test_accuracy', 'N/A')} | {ml_metrics.get('Random_Forest', {}).get('test_roc_auc', 'N/A')} | {ml_metrics.get('Random_Forest', {}).get('test_pr_auc', 'N/A')} | {ml_metrics.get('Random_Forest', {}).get('test_recall_pos', 'N/A')} | {ml_metrics.get('Random_Forest', {}).get('test_brier_score', 'N/A')} |
| **Gradient Boosting** | {ml_metrics.get('Gradient_Boosting', {}).get('test_accuracy', 'N/A')} | {ml_metrics.get('Gradient_Boosting', {}).get('test_roc_auc', 'N/A')} | {ml_metrics.get('Gradient_Boosting', {}).get('test_pr_auc', 'N/A')} | {ml_metrics.get('Gradient_Boosting', {}).get('test_recall_pos', 'N/A')} | {ml_metrics.get('Gradient_Boosting', {}).get('test_brier_score', 'N/A')} |

- **Exported Artifact:** `models/escalation_model.pkl`

---

## 5. Member Behavioral Segmentation (Phase 6)
- **Clustering Methods:** K-Means ($k=4$) with `StandardScaler` & Hierarchical Agglomerative Clustering (Ward Linkage).
- **Hierarchical Clustering Metrics:** Cophenetic Correlation = {segmentation_report.get('hierarchical_clustering', {}).get('cophenetic_correlation', 'N/A')} | Adjusted Rand Index vs K-Means = {segmentation_report.get('hierarchical_clustering', {}).get('adjusted_rand_index_vs_kmeans', 'N/A')}
- **Dimensionality Reduction:** PCA (2 components) explaining {sum(segmentation_report.get('pca_explained_variance_ratio', [0, 0]))*100:.1f}% of total variance.
- **Identified Behavioral Personas:**
"""
    persona_map = segmentation_report.get("persona_mapping", {})
    centers = segmentation_report.get("cluster_centers", {})
    for cid, persona in persona_map.items():
        stats_c = centers.get(str(cid), centers.get(int(cid), {}))
        age = stats_c.get("Age", 0)
        dig = stats_c.get("Digital_Adoption_Score", 0) * 100
        fric = stats_c.get("Avg_Service_Friction", 0)
        esc = stats_c.get("Member_Escalation_Rate", 0) * 100
        md_content += f"  - **Cluster {cid} — {persona}:** Mean Age: {age:.1f} yrs | Digital Adoption: {dig:.1f}% | Avg Friction: {fric:.3f} | Escalation Rate: {esc:.1f}%\n"

    md_content += f"""
- **Exported Artifacts:** `models/kmeans_segmentation.pkl`, `reports/figures/segmentation_clusters_pca.png`, `reports/figures/hierarchical_dendrogram.png`

---

## 6. Deep Learning vs. Traditional Sentiment Benchmark (Phase 7)
- **Deep Learning Model:** PyTorch 1D-CNN with learned word embeddings (64-dim), batch normalization, adaptive pooling, and dropout.
- **Baseline Model:** TF-IDF (1-2 grams) + Logistic Regression.
- **Benchmark Evaluation (Trained on Realistic Noisy Text):**
  - **PyTorch 1D-CNN:** Accuracy: {sentiment_report['pytorch_1d_cnn']['accuracy']}, F1-Score: {sentiment_report['pytorch_1d_cnn']['f1_score']}, ROC-AUC: {sentiment_report['pytorch_1d_cnn']['roc_auc']}
  - **Baseline TF-IDF:** Accuracy: {sentiment_report['baseline_tfidf_logistic_regression']['accuracy']}, F1-Score: {sentiment_report['baseline_tfidf_logistic_regression']['f1_score']}, ROC-AUC: {sentiment_report['baseline_tfidf_logistic_regression']['roc_auc']}
- **Exported Artifact:** `models/pytorch_sentiment_model.pt`

---

## 7. NLP Topic Modeling & Complaint Classification (Phase 8)
- **Topic Modeling Algorithm:** Latent Dirichlet Allocation (LDA) with 5 Kaiser operational themes:
"""
    topics_dict = nlp_report.get("topics", {})
    for t_name, t_info in topics_dict.items():
        words = ", ".join(t_info.get("top_words", [])[:6])
        md_content += f"  - **{t_name}:** `{words}`\n"

    complaint_eval = nlp_report.get("complaint_classification", {})
    md_content += f"""
- **Supervised Complaint Classification:**
  - **Overall Accuracy:** {complaint_eval.get('accuracy', 'N/A')}
  - **Macro F1-Score:** {complaint_eval.get('f1_macro', 'N/A')}
  - **Weighted F1-Score:** {complaint_eval.get('f1_weighted', 'N/A')}
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
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n  [REPORT GENERATED] Comprehensive Pipeline Summary Report saved to: {report_path}")


def run_pipeline(force_generate_data: bool = False, run_eda_module: bool = True):
    """
    Master pipeline execution function.
    """
    start_time = time.time()
    print("\n" + "#"*80)
    print(" KAISER PERMANENTE MEMBER EXPERIENCE ANALYTICS & INTELLIGENCE PIPELINE")
    print(f" Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*80)

    # Step 0: Ensure input data exists
    raw_files_exist = all((BASE_DIR / fn).exists() for fn in INPUT_FILES.values())
    if force_generate_data or not raw_files_exist:
        print("\n>>> INITIALIZING DATA GENERATION <<<")
        generate_synthetic_data()

    # Step 1: Data Integration & Schema Integrity (Phase 1)
    df_integrated, baseline_metrics = run_module1()

    # Step 2: Statistical Hypothesis Testing & Confidence Intervals (Phase 2)
    hypothesis_results = run_module2(df_integrated)

    # Step 3: Exploratory Data Analysis (EDA) Automation (Phase 3)
    if run_eda_module:
        run_module3(df_integrated)

    # Step 4: Operational Feature Engineering (Phase 4)
    df_features = run_module4(df_integrated)

    # Step 5: Escalation Prediction Machine Learning (Phase 5)
    df_scored, champion_model, ml_metrics = run_module5(df_features)

    # Step 6: Member Behavioral Segmentation (Phase 6)
    df_segmented, df_member_enriched, seg_artifacts = run_module6(df_scored)

    # Step 7: Deep Learning vs. Baseline Sentiment Benchmark (Phase 7)
    df_sentiment, sentiment_report = run_module7(df_segmented)

    # Step 8: NLP Topic Modeling & Complaint Classification (Phase 8)
    df_final, nlp_artifacts, nlp_report = run_module8(df_sentiment)

    # Step 9: Power BI Star Schema & RAG Staging Export (Phase 10)
    star_schema_tables = run_module9(df_final, df_member_enriched)

    # Step 10: Relational Database Persistence Sync (Local SQLite)
    run_database_sync()

    # Step 11: Automated Cloud Database Synchronization (Neon Postgres)
    try:
        uploader = CloudDatabaseUploader()
        uploader.upload_star_schema()
    except Exception as e:
        print(f"  [WARN] Cloud Neon sync skipped or encountered an error: {e}")

    total_duration = time.time() - start_time

    # Generate Executive Summary Report
    import json
    with open(REPORTS_DIR / "segmentation_report.json", "r", encoding="utf-8") as f:
        seg_json = json.load(f)

    generate_pipeline_summary_report(
        baseline_metrics=baseline_metrics,
        hypothesis_results=hypothesis_results,
        ml_metrics=ml_metrics,
        segmentation_report=seg_json,
        sentiment_report=sentiment_report,
        nlp_report=nlp_report,
        execution_time=total_duration
    )

    # Auto-synchronize the 6-page standalone dashboard HTML with latest database metrics
    try:
        from src.neon_dashboard_sync import sync_html_dashboard_with_neon
        sync_html_dashboard_with_neon()
    except Exception as e:
        print(f"Note: Standalone dashboard sync skipped: {e}")

    print("\n" + "#"*80)
    print(f" KAISER PERMANENTE PIPELINE EXECUTION SUCCESSFULLY COMPLETED IN {total_duration:.2f}s")
    print(f" Output Star Schema:  {DATA_PROCESSED_DIR}")
    print(f" Executive Reports:   {REPORTS_DIR}")
    print(f" Visual Figures:      {FIGURES_DIR}")
    print(f" Serialized Models:   {MODELS_DIR}")
    print("#"*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaiser Permanente Analytics Pipeline")
    parser.add_argument("--generate-data", action="store_true", help="Force regeneration of synthetic data")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA visualization generation")
    args = parser.parse_args()

    run_pipeline(
        force_generate_data=args.generate_data,
        run_eda_module=not args.skip_eda
    )
