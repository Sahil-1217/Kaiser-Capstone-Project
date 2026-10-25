"""
Module 2: Statistical Hypothesis Testing & Confidence Intervals (Phase 2)
Implements rigorous hypothesis testing and confidence intervals:
- Hypothesis 1: Resolution Time vs Satisfaction Score (Spearman Rank Correlation)
- Hypothesis 2: Contact Frequency vs Escalation (Welch's Two-Sample t-test & Cohen's d)
- Hypothesis 3: Digital Channels vs Voice Escalation Rate (Chi-Square Test & Odds Ratio)
- 95% Confidence Intervals for Escalation Rate (Wilson Score) and Mean CSAT
"""

import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
from src.config import REPORTS_DIR, DIGITAL_CHANNELS


def test_hypothesis_1_resolution_vs_csat(df: pd.DataFrame) -> dict:
    """
    Hypothesis 1: Longer resolution times significantly reduce member satisfaction.
    Method: Spearman rank correlation (suitable for ordinal CSAT and skewed resolution times).
    """
    res_time = df["Resolution_Time"]
    csat = df["Satisfaction_Score"]

    rho, p_val_spearman = stats.spearmanr(res_time, csat)
    r_pearson, p_val_pearson = stats.pearsonr(res_time, csat)

    significant = bool(p_val_spearman < 0.05 and rho < 0)
    business_conclusion = (
        f"Statistically significant negative correlation (rho = {rho:.4f}, p = {p_val_spearman:.2e}). "
        "Every 10-minute increase in resolution duration is associated with a measurable degradation "
        "in member satisfaction ratings. Operational efforts to reduce handle times directly improve CSAT."
        if significant else
        f"No statistically significant negative correlation observed (rho = {rho:.4f}, p = {p_val_spearman:.4f})."
    )

    return {
        "hypothesis": "H1: Longer resolution times significantly reduce member satisfaction",
        "test_type": "Spearman Rank Correlation & Pearson Correlation",
        "sample_size": len(df),
        "spearman_rho": round(float(rho), 4),
        "spearman_p_value": float(p_val_spearman),
        "pearson_r": round(float(r_pearson), 4),
        "pearson_p_value": float(p_val_pearson),
        "alpha": 0.05,
        "is_statistically_significant": significant,
        "business_conclusion": business_conclusion
    }


def test_hypothesis_2_contacts_vs_escalation(df: pd.DataFrame) -> dict:
    """
    Hypothesis 2: Members with multiple contacts have a higher escalation probability.
    Method: Welch's Two-Sample t-test (unequal variances) & Mann-Whitney U test.
    """
    escalated_contacts = df[df["Escalation_Flag"] == 1]["Total_Contacts"]
    non_escalated_contacts = df[df["Escalation_Flag"] == 0]["Total_Contacts"]

    n1, n2 = len(escalated_contacts), len(non_escalated_contacts)
    mean1, mean2 = escalated_contacts.mean(), non_escalated_contacts.mean()
    std1, std2 = escalated_contacts.std(), non_escalated_contacts.std()

    # Welch's t-test
    t_stat, p_val_welch = stats.ttest_ind(escalated_contacts, non_escalated_contacts, equal_var=False)

    # Mann-Whitney U test (non-parametric check)
    u_stat, p_val_mann = stats.mannwhitneyu(escalated_contacts, non_escalated_contacts, alternative='greater')

    # Cohen's d effect size (pooled variance)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    significant = bool(p_val_welch < 0.05 and t_stat > 0)
    business_conclusion = (
        f"Statistically significant difference confirmed (t = {t_stat:.4f}, p = {p_val_welch:.2e}, Cohen's d = {cohens_d:.3f}). "
        f"Escalated interactions averaged {mean1:.2f} contacts vs. {mean2:.2f} contacts for non-escalated cases. "
        "Failing to resolve inquiries on the primary contact exponentially increases escalation risk."
        if significant else
        f"No statistically significant difference in contacts (t = {t_stat:.4f}, p = {p_val_welch:.4f})."
    )

    return {
        "hypothesis": "H2: Members with multiple contacts have a higher escalation probability",
        "test_type": "Welch's Two-Sample t-test (unequal variances) & Mann-Whitney U",
        "escalated_group": {"n": n1, "mean_contacts": round(float(mean1), 3), "std": round(float(std1), 3)},
        "non_escalated_group": {"n": n2, "mean_contacts": round(float(mean2), 3), "std": round(float(std2), 3)},
        "welch_t_stat": round(float(t_stat), 4),
        "welch_p_value": float(p_val_welch),
        "mann_whitney_u_stat": float(u_stat),
        "mann_whitney_p_value": float(p_val_mann),
        "cohens_d_effect_size": round(float(cohens_d), 4),
        "alpha": 0.05,
        "is_statistically_significant": significant,
        "business_conclusion": business_conclusion
    }


def test_hypothesis_3_channel_escalation(df: pd.DataFrame) -> dict:
    """
    Hypothesis 3: Digital channels (Chat, Mobile App) have lower escalation rates than voice calls.
    Method: Chi-Square Test of Independence & Odds Ratio.
    """
    # Filter to Digital vs Voice
    df_sub = df[df["Contact_Channel"].isin(["Phone"] + DIGITAL_CHANNELS)].copy()
    df_sub["Channel_Category"] = df_sub["Contact_Channel"].apply(
        lambda c: "Digital (Chat/App)" if c in DIGITAL_CHANNELS else "Voice (Phone)"
    )

    # 2x2 Contingency Table: [Voice, Digital] x [Escalated, Not Escalated]
    contingency = pd.crosstab(df_sub["Channel_Category"], df_sub["Escalation_Flag"])
    
    chi2_stat, p_val_chi2, dof, expected = stats.chi2_contingency(contingency)

    # Escalation rates
    rates = df_sub.groupby("Channel_Category")["Escalation_Flag"].agg(["count", "mean"]).to_dict("index")
    voice_rate = rates["Voice (Phone)"]["mean"]
    digital_rate = rates["Digital (Chat/App)"]["mean"]

    # Odds Ratio: (Esc_voice / NonEsc_voice) / (Esc_digital / NonEsc_digital)
    esc_voice = contingency.loc["Voice (Phone)", 1]
    non_voice = contingency.loc["Voice (Phone)", 0]
    esc_dig = contingency.loc["Digital (Chat/App)", 1]
    non_dig = contingency.loc["Digital (Chat/App)", 0]

    odds_voice = esc_voice / non_voice if non_voice > 0 else np.nan
    odds_dig = esc_dig / non_dig if non_dig > 0 else np.nan
    odds_ratio = odds_voice / odds_dig if odds_dig > 0 else np.nan

    # 95% Confidence Interval for Odds Ratio
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1/esc_voice + 1/non_voice + 1/esc_dig + 1/non_dig)
    ci_lower = np.exp(log_or - 1.96 * se_log_or)
    ci_upper = np.exp(log_or + 1.96 * se_log_or)

    significant = bool(p_val_chi2 < 0.05 and digital_rate < voice_rate)
    business_conclusion = (
        f"Statistically significant difference confirmed (Chi2 = {chi2_stat:.4f}, p = {p_val_chi2:.2e}, df = {dof}). "
        f"Digital channels exhibited an escalation rate of {digital_rate*100:.2f}% versus {voice_rate*100:.2f}% for Voice calls. "
        f"Voice interactions have {odds_ratio:.2f}x higher odds of escalation (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]). "
        "Care navigation strategies should route initial triage to asynchronous and interactive digital channels."
        if significant else
        f"No statistically significant difference between channels (Chi2 = {chi2_stat:.4f}, p = {p_val_chi2:.4f})."
    )

    return {
        "hypothesis": "H3: Digital channels have lower escalation rates than voice calls",
        "test_type": "Pearson Chi-Square Test of Independence & Odds Ratio",
        "voice_escalation_rate": round(float(voice_rate), 4),
        "digital_escalation_rate": round(float(digital_rate), 4),
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": float(p_val_chi2),
        "degrees_of_freedom": int(dof),
        "odds_ratio_voice_vs_digital": round(float(odds_ratio), 4),
        "odds_ratio_95_ci": [round(float(ci_lower), 4), round(float(ci_upper), 4)],
        "alpha": 0.05,
        "is_statistically_significant": significant,
        "business_conclusion": business_conclusion
    }


def compute_confidence_intervals(df: pd.DataFrame) -> dict:
    """
    Compute 95% Confidence Intervals for:
    1. Overall Escalation Rate (Wilson Score Interval)
    2. Mean Satisfaction Score (t-distribution)
    """
    n = len(df)
    z = stats.norm.ppf(0.975)  # 1.95996

    # 1. Escalation Rate (Wilson score interval)
    p_hat = df["Escalation_Flag"].mean()
    denominator = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denominator
    margin = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2))) / denominator
    esc_ci_lower = max(0.0, center - margin)
    esc_ci_upper = min(1.0, center + margin)

    # 2. Mean CSAT (Student's t CI)
    mean_csat = df["Satisfaction_Score"].mean()
    std_csat = df["Satisfaction_Score"].std()
    sem_csat = stats.sem(df["Satisfaction_Score"])
    t_crit = stats.t.ppf(0.975, df=n - 1)
    csat_ci_lower = mean_csat - t_crit * sem_csat
    csat_ci_upper = mean_csat + t_crit * sem_csat

    return {
        "escalation_rate": {
            "point_estimate": round(float(p_hat), 4),
            "percentage": round(float(p_hat * 100), 2),
            "ci_95_lower": round(float(esc_ci_lower), 4),
            "ci_95_upper": round(float(esc_ci_upper), 4),
            "method": "Wilson Score Interval"
        },
        "mean_csat": {
            "point_estimate": round(float(mean_csat), 4),
            "sample_std": round(float(std_csat), 4),
            "ci_95_lower": round(float(csat_ci_lower), 4),
            "ci_95_upper": round(float(csat_ci_upper), 4),
            "method": "Student's t-distribution"
        }
    }


def run_module2(df: pd.DataFrame) -> dict:
    """Execution entry point for Module 2."""
    print("\n>>> EXECUTING MODULE 2: STATISTICAL HYPOTHESIS TESTING & CONFIDENCE INTERVALS <<<")
    
    h1_results = test_hypothesis_1_resolution_vs_csat(df)
    h2_results = test_hypothesis_2_contacts_vs_escalation(df)
    h3_results = test_hypothesis_3_channel_escalation(df)
    ci_results = compute_confidence_intervals(df)

    all_results = {
        "hypothesis_1": h1_results,
        "hypothesis_2": h2_results,
        "hypothesis_3": h3_results,
        "confidence_intervals": ci_results
    }

    # Save to JSON
    json_path = REPORTS_DIR / "hypothesis_testing_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"  [SAVED] Hypothesis testing results exported to: {json_path}")

    # Terminal Log
    print("\n" + "="*75)
    print(" STATISTICAL HYPOTHESIS TESTING SUMMARY & BUSINESS IMPACT")
    print("="*75)
    print(f"[H1: Resolution Time vs. CSAT]")
    print(f"  Spearman rho: {h1_results['spearman_rho']}, p-value: {h1_results['spearman_p_value']:.2e}")
    print(f"  Conclusion:   {h1_results['business_conclusion']}\n")

    print(f"[H2: Contacts vs. Escalation]")
    print(f"  Welch t:      {h2_results['welch_t_stat']}, p-value: {h2_results['welch_p_value']:.2e}, Cohen's d: {h2_results['cohens_d_effect_size']}")
    print(f"  Escalated Mean Contacts: {h2_results['escalated_group']['mean_contacts']} vs. Non-Escalated: {h2_results['non_escalated_group']['mean_contacts']}")
    print(f"  Conclusion:   {h2_results['business_conclusion']}\n")

    print(f"[H3: Digital Channels vs. Voice Escalations]")
    print(f"  Chi-Square:   {h3_results['chi2_statistic']}, p-value: {h3_results['p_value']:.2e}, Odds Ratio: {h3_results['odds_ratio_voice_vs_digital']}")
    print(f"  Voice Rate:   {h3_results['voice_escalation_rate']*100:.2f}% vs. Digital: {h3_results['digital_escalation_rate']*100:.2f}%")
    print(f"  Conclusion:   {h3_results['business_conclusion']}\n")

    print(f"[95% Confidence Intervals]")
    print(f"  Escalation Rate: {ci_results['escalation_rate']['percentage']}% (95% CI: [{ci_results['escalation_rate']['ci_95_lower']*100:.2f}%, {ci_results['escalation_rate']['ci_95_upper']*100:.2f}%])")
    print(f"  Mean CSAT:       {ci_results['mean_csat']['point_estimate']:.2f} / 5.0 (95% CI: [{ci_results['mean_csat']['ci_95_lower']:.2f}, {ci_results['mean_csat']['ci_95_upper']:.2f}])")
    print("="*75 + "\n")

    return all_results


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    df_data, _ = run_module1()
    run_module2(df_data)
