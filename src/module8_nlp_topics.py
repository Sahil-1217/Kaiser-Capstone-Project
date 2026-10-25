"""
Module 8: NLP Topic Modeling & Complaint Classification (Phase 8)
Cleans transcripts, extracts high-friction TF-IDF keywords, trains an LDA topic model
for 5 core Kaiser Permanente operational themes (Appointments, Billing, Claims, Pharmacy, Provider Access),
and appends Dominant_Topic back to each interaction record.
"""

import os
import sys
import re
import json
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, NLP_THEMES

# Domain stop words and dialogue conversational fillers
STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself",
    "yourselves", "agent", "member", "hello", "hi", "user", "bot", "yes", "please", "thank", "thanks",
    "kaiser", "permanente", "call", "chat", "inquiry", "help", "need", "let", "see", "get", "got", "know"
])


def clean_text(text: str) -> str:
    """
    Cleans transcript text: lowercase, punctuation removal, stopword removal,
    and suffix-based lemmatization for common healthcare nouns.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    cleaned_words = []
    for w in words:
        if len(w) < 3 or w in STOP_WORDS:
            continue
        # Rule-based lemmatization for common healthcare plurals and verb inflections
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("es") and len(w) > 4 and not w.endswith("tes"):
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        elif w.endswith("ing") and len(w) > 5:
            w = w[:-3]
        elif w.endswith("ed") and len(w) > 4:
            w = w[:-2]
        
        if w not in STOP_WORDS and len(w) >= 3:
            cleaned_words.append(w)

    return " ".join(cleaned_words)


def extract_high_friction_keywords(df: pd.DataFrame, top_n: int = 15) -> dict:
    """
    Uses TF-IDF to compare terms that disproportionately appear in escalated interactions.
    """
    df_esc = df[df["Escalation_Flag"] == 1]
    df_non = df[df["Escalation_Flag"] == 0]

    texts_esc = df_esc["Cleaned_Transcript"]
    texts_non = df_non["Cleaned_Transcript"]

    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    
    # Fit on all, transform separately
    tfidf.fit(df["Cleaned_Transcript"])
    words = np.array(tfidf.get_feature_names_out())

    esc_mean = np.asarray(tfidf.transform(texts_esc).mean(axis=0)).flatten()
    non_mean = np.asarray(tfidf.transform(texts_non).mean(axis=0)).flatten()

    # Friction lift: esc_mean - non_mean
    friction_lift = esc_mean - non_mean
    top_friction_indices = np.argsort(friction_lift)[::-1][:top_n]

    high_friction_terms = {
        words[i]: {
            "friction_lift": round(float(friction_lift[i]), 4),
            "escalated_tfidf": round(float(esc_mean[i]), 4),
            "non_escalated_tfidf": round(float(non_mean[i]), 4)
        }
        for i in top_friction_indices
    }

    print("\n  [HIGH-FRICTION KEYWORDS (Escalated vs. Non-Escalated)]")
    for term, data in list(high_friction_terms.items())[:8]:
        print(f"    * '{term:<20}': Escalated TF-IDF = {data['escalated_tfidf']:.4f} (Lift: +{data['friction_lift']:.4f})")

    return high_friction_terms


def train_lda_topic_model(corpus: list, n_topics: int = 5):
    """
    Fits LDA Topic Model and maps topics to 5 core Kaiser operational themes.
    """
    print(f"\n  [TOPIC MODELING] Fitting Latent Dirichlet Allocation (LDA) with {n_topics} topics...")
    vectorizer = CountVectorizer(max_features=2500, min_df=3, max_df=0.85)
    dtm = vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=25,
        learning_method="batch",
        n_jobs=-1
    )
    doc_topic_dist = lda.fit_transform(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords = {}
    assigned_theme_names = {}

    theme_keyword_signals = {
        "Appointments": ["appointment", "schedule", "reschedule", "slot", "physician", "doctor", "triage", "walk", "visit"],
        "Billing": ["bill", "deductible", "copay", "statement", "payment", "charge", "balance", "cost", "maximum"],
        "Claims": ["claim", "denied", "reimbursement", "appeal", "emergency", "eob", "adjudication", "out", "network"],
        "Pharmacy": ["prescription", "refill", "pharmacy", "medication", "delivery", "insulin", "order", "mail", "drug"],
        "Provider Access": ["specialist", "referral", "provider", "therapist", "mental", "health", "pediatrician", "network", "access"]
    }

    # Match each LDA topic to one of the 5 Kaiser themes
    used_themes = set()
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-12:-1]
        top_words = [feature_names[i] for i in top_indices]
        topic_keywords[topic_idx] = top_words

        # Score alignment with Kaiser operational themes
        best_theme = None
        best_score = -1
        for theme, signals in theme_keyword_signals.items():
            if theme in used_themes:
                continue
            score = sum(1 for w in top_words if w in signals)
            if score > best_score:
                best_score = score
                best_theme = theme

        if not best_theme:
            # Fallback
            remaining = [t for t in NLP_THEMES if t not in used_themes]
            best_theme = remaining[0] if remaining else f"Theme_{topic_idx}"

        used_themes.add(best_theme)
        assigned_theme_names[topic_idx] = best_theme

        print(f"    Topic {topic_idx} -> '{best_theme}': {', '.join(top_words[:6])}")

    return lda, vectorizer, doc_topic_dist, topic_keywords, assigned_theme_names


def plot_topic_distributions(df: pd.DataFrame):
    """Plots interaction volume and escalation rate per dominant topic."""
    topic_stats = df.groupby("Dominant_Topic").agg(
        Volume=("Interaction_ID", "count"),
        Escalation_Rate=("Escalation_Flag", "mean"),
        Mean_Resolution_Time=("Resolution_Time", "mean"),
        Mean_CSAT=("Satisfaction_Score", "mean")
    ).sort_values("Volume", ascending=False).reset_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(topic_stats))
    width = 0.40

    bars1 = ax1.bar(x - width/2, topic_stats["Volume"], width, label="Interaction Volume", color="#3498db", edgecolor="black")
    ax1.set_ylabel("Interaction Volume", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(topic_stats["Dominant_Topic"], fontsize=10, fontweight="bold")
    ax1.set_title("Kaiser Permanente Topic Classification: Interaction Volume & Escalation Rate", fontsize=13, fontweight="bold", pad=14)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, topic_stats["Escalation_Rate"] * 100, width, label="Escalation Rate (%)", color="#e74c3c", edgecolor="black")
    ax2.set_ylabel("Escalation Rate (%)", fontsize=11, fontweight="bold", color="#e74c3c")
    ax2.set_ylim(0, 50)
    ax2.grid(False)

    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width()/2, h + 80, f"{int(h):,}", ha="center", fontsize=9, fontweight="bold")
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2, h + 1.0, f"{h:.1f}%", ha="center", fontsize=9, fontweight="bold", color="#c0392b")

    plt.tight_layout()
    output_path = FIGURES_DIR / "topic_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")


def run_module8(df: pd.DataFrame) -> tuple:
    """Execution entry point for Module 8."""
    print("\n>>> EXECUTING MODULE 8: NLP TOPIC MODELING & COMPLAINT CLASSIFICATION <<<")
    df = df.copy()

    # Combine available text: Call_Transcript, Chat_Transcript, Feedback_Comments
    print("  [TEXT PREPROCESSING] Cleaning and lemmatizing transcript text corpus...")
    combined_text = (
        df["Call_Transcript"].fillna("") + " " +
        df["Chat_Transcript"].fillna("") + " " +
        df["Feedback_Comments"].fillna("")
    )
    df["Cleaned_Transcript"] = combined_text.apply(clean_text)

    # 1. High-Friction Keyword Extraction
    high_friction_terms = extract_high_friction_keywords(df)

    # 2. LDA Topic Modeling
    lda, vectorizer, doc_topic_dist, topic_keywords, assigned_theme_names = train_lda_topic_model(
        df["Cleaned_Transcript"].tolist(), n_topics=5
    )

    # Assign Dominant Topic
    dominant_topic_indices = np.argmax(doc_topic_dist, axis=1)
    df["Dominant_Topic_Index"] = dominant_topic_indices
    df["Dominant_Topic"] = [assigned_theme_names[idx] for idx in dominant_topic_indices]
    df["Topic_Confidence"] = np.round(np.max(doc_topic_dist, axis=1), 4)

    print("\n  [DOMINANT TOPIC DISTRIBUTION]")
    top_dist = df["Dominant_Topic"].value_counts()
    for topic, count in top_dist.items():
        print(f"    * {topic:<20}: {count:>5} ({count/len(df)*100:5.1f}%)")

    # 3. Topic Visualizations
    plot_topic_distributions(df)

    # 4. Save Model Artifacts
    nlp_artifacts = {
        "lda_model": lda,
        "vectorizer": vectorizer,
        "assigned_theme_names": assigned_theme_names,
        "topic_keywords": topic_keywords
    }
    model_path = MODELS_DIR / "lda_topic_model.pkl"
    joblib.dump(nlp_artifacts, model_path)
    print(f"  [MODEL EXPORT] Saved LDA topic model to: {model_path}")

    # Export Topic Report
    nlp_report = {
        "topics": {
            assigned_theme_names[idx]: {
                "topic_index": idx,
                "top_words": words
            }
            for idx, words in topic_keywords.items()
        },
        "high_friction_keywords": high_friction_terms
    }
    report_path = REPORTS_DIR / "nlp_topic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(nlp_report, f, indent=4)
    print(f"  [REPORT SAVED] Exported NLP topic report to: {report_path}")

    return df, nlp_artifacts, nlp_report


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    from src.module4_feature_engineering import run_module4
    from src.module5_escalation_ml import run_module5
    from src.module6_segmentation import run_module6
    from src.module7_sentiment_dl import run_module7

    df_data, _ = run_module1()
    df_feat = run_module4(df_data)
    df_scored, _, _ = run_module5(df_feat)
    df_seg, _, _ = run_module6(df_scored)
    df_sent, _ = run_module7(df_seg)
    df_final, _, _ = run_module8(df_sent)
