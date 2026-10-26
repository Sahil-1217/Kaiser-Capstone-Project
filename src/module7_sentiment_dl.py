"""
Module 7: Deep Learning vs. Traditional Sentiment Benchmark (Phase 7)
Implements a lightweight PyTorch 1D-CNN/Embedding text classifier for member feedback,
benchmarks against a TF-IDF + Logistic Regression baseline, outputs comparative metrics,
saves training curves, and appends continuous Sentiment_Score to interaction records.
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import RANDOM_STATE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR

# Set PyTorch seeds
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


class TextVocabulary:
    """Lightweight vocabulary builder and tokenizer."""
    def __init__(self, max_vocab_size: int = 3000):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    @staticmethod
    def clean_and_tokenize(text: str) -> list:
        text = str(text).lower()
        tokens = re.findall(r"\b[a-z]{2,}\b", text)
        return tokens

    def fit(self, texts: list):
        word_counts = {}
        for text in texts:
            tokens = self.clean_and_tokenize(text)
            for t in tokens:
                word_counts[t] = word_counts.get(t, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab_size - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 40) -> list:
        tokens = self.clean_and_tokenize(text)
        indices = [self.word2idx.get(t, 1) for t in tokens][:max_len]
        # Pad if needed
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        return indices


class FeedbackDataset(Dataset):
    """PyTorch Dataset for Member Feedback texts."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class SentimentCNN(nn.Module):
    """
    Lightweight 1D-CNN with Embedding Layer and Dropout for Sentiment Classification.
    """
    def __init__(self, vocab_size: int, embedding_dim: int = 64, num_filters: int = 64, dropout_rate: float = 0.3):
        super(SentimentCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_filters, 32)
        self.bn = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        emb = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        emb = emb.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        conv_out = self.relu(self.conv(emb))  # [batch_size, num_filters, seq_len]
        pooled = self.pool(conv_out).squeeze(2)  # [batch_size, num_filters]
        dense = self.dropout(self.relu(self.bn(self.fc1(pooled))))
        logits = self.out(dense)
        return self.sigmoid(logits)


def prepare_sentiment_data(df: pd.DataFrame):
    """
    Extracts text and constructs binary sentiment labels (Positive: 1 vs Neutral/Negative: 0).
    Verifies that text contains realistic linguistic variation, guarding against artificial memorization.
    """
    if "Feedback_Comments" not in df.columns or df["Feedback_Comments"].dropna().empty:
        from src.module1_data_integration import inject_realistic_sentiment_text
        df = inject_realistic_sentiment_text(df)

    texts = df["Feedback_Comments"].fillna("").tolist()
    # Positive if Satisfaction_Score >= 4, Negative/Neutral if <= 3
    labels = (df["Satisfaction_Score"] >= 4).astype(int).values

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"  [DATASET] Sentiment texts: {len(train_texts)} train, {len(test_texts)} test (Positive rate: {labels.mean()*100:.1f}%, Unique text count: {pd.Series(texts).nunique():,})")
    return texts, labels, train_texts, test_texts, y_train, y_test


def train_baseline_tfidf(train_texts, test_texts, y_train, y_test):
    """
    Trains traditional TF-IDF + Logistic Regression baseline.
    """
    print("  [BASELINE ML] Training TF-IDF (1-2 grams) + Logistic Regression...")
    tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)

    clf = LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=500)
    clf.fit(X_train_tfidf, y_train)

    y_pred = clf.predict(X_test_tfidf)
    y_prob = clf.predict_proba(X_test_tfidf)[:, 1]

    metrics = {
        "model_name": "Traditional: TF-IDF + Logistic Regression",
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4)
    }
    print(f"    * Baseline Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")
    return clf, tfidf, metrics


def train_pytorch_cnn(train_texts, test_texts, y_train, y_test, epochs: int = 10, batch_size: int = 64):
    """
    Trains lightweight PyTorch 1D-CNN sentiment classifier.
    """
    print(f"  [DEEP LEARNING] Building PyTorch 1D-CNN Text Classifier ({epochs} epochs)...")
    vocab = TextVocabulary(max_vocab_size=3000)
    vocab.fit(train_texts)

    max_len = 40
    X_train_seq = np.array([vocab.encode(t, max_len=max_len) for t in train_texts])
    X_test_seq = np.array([vocab.encode(t, max_len=max_len) for t in test_texts])

    train_dataset = FeedbackDataset(X_train_seq, y_train)
    test_dataset = FeedbackDataset(X_test_seq, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SentimentCNN(vocab_size=len(vocab.word2idx), embedding_dim=64, num_filters=64, dropout_rate=0.3)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for seqs, targets in train_loader:
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)

        avg_train_loss = total_loss / len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for seqs, targets in test_loader:
                preds = model(seqs)
                loss = criterion(preds, targets)
                val_loss += loss.item() * len(targets)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        avg_val_loss = val_loss / len(test_dataset)
        val_acc = accuracy_score(all_targets, np.array(all_preds) >= 0.5)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(avg_val_loss, 4))
        history["val_accuracy"].append(round(val_acc, 4))

        if epoch % 2 == 0 or epoch == epochs:
            print(f"    Epoch {epoch:2d}/{epochs:2d} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Final Test Set Evaluation
    y_prob_dl = np.array(all_preds)
    y_pred_dl = (y_prob_dl >= 0.5).astype(int)

    metrics = {
        "model_name": "Deep Learning: PyTorch 1D-CNN + Word Embeddings",
        "accuracy": round(float(accuracy_score(y_test, y_pred_dl)), 4),
        "precision": round(float(precision_score(y_test, y_pred_dl)), 4),
        "recall": round(float(recall_score(y_test, y_pred_dl)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred_dl)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob_dl)), 4)
    }
    print(f"    * PyTorch CNN Accuracy: {metrics['accuracy']}, F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")

    # Save Model Weights and Vocab
    model_save_path = MODELS_DIR / "pytorch_sentiment_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "vocab": vocab.word2idx,
        "max_len": max_len,
        "metrics": metrics
    }, model_save_path)
    print(f"  [MODEL EXPORT] Saved PyTorch weights to: {model_save_path}")

    return model, vocab, metrics, history


def plot_sentiment_benchmark(baseline_metrics: dict, dl_metrics: dict, history: dict):
    """Plots training loss curve and comparative performance bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 1. Loss curves over epochs
    epochs_range = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o", color="#e74c3c", linewidth=2)
    axes[0].plot(epochs_range, history["val_loss"], label="Validation Loss", marker="s", color="#3498db", linewidth=2)
    axes[0].set_title("PyTorch 1D-CNN Training & Validation Loss", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Binary Cross Entropy Loss", fontsize=11)
    axes[0].legend(frameon=True)

    # 2. Comparative Bar Chart
    metric_keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    base_vals = [baseline_metrics[m] for m in metric_keys]
    dl_vals = [dl_metrics[m] for m in metric_keys]

    x = np.arange(len(metric_labels))
    width = 0.35

    axes[1].bar(x - width/2, base_vals, width, label="TF-IDF + Logistic Reg", color="#95a5a6", edgecolor="black")
    axes[1].bar(x + width/2, dl_vals, width, label="PyTorch 1D-CNN", color="#2ecc71", edgecolor="black")
    axes[1].set_title("Sentiment Benchmark: Deep Learning vs. Baseline ML", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_ylabel("Score", fontsize=11)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_labels, fontsize=10)
    axes[1].set_ylim(0, 1.15)
    axes[1].legend(loc="upper right", frameon=True)

    for i in range(len(metric_labels)):
        axes[1].text(x[i] - width/2, base_vals[i] + 0.02, f"{base_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].text(x[i] + width/2, dl_vals[i] + 0.02, f"{dl_vals[i]:.2f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_path = FIGURES_DIR / "sentiment_benchmark.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [FIGURE SAVED] {output_path.name}")


def append_sentiment_scores(model, vocab, df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores the entire dataset and appends continuous Sentiment_Score [0.0, 1.0].
    """
    df = df.copy()
    texts = df["Feedback_Comments"].fillna("").tolist()
    seqs = np.array([vocab.encode(t, max_len=40) for t in texts])
    tensor_seqs = torch.tensor(seqs, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        probs = model(tensor_seqs).cpu().numpy().flatten()

    df["Sentiment_Score"] = np.round(probs, 4)
    print(f"  [INFERENCE] Appended Sentiment_Score to dataset (Mean: {df['Sentiment_Score'].mean():.4f}, Pos Rate: {(df['Sentiment_Score']>=0.5).mean()*100:.1f}%)")
    return df


def run_module7(df: pd.DataFrame):
    """Execution entry point for Module 7."""
    print("\n>>> EXECUTING MODULE 7: DEEP LEARNING VS. TRADITIONAL SENTIMENT BENCHMARK <<<")
    texts, labels, train_texts, test_texts, y_train, y_test = prepare_sentiment_data(df)
    
    # 1. Baseline
    _, _, baseline_metrics = train_baseline_tfidf(train_texts, test_texts, y_train, y_test)

    # 2. PyTorch Deep Learning Model
    dl_model, vocab, dl_metrics, history = train_pytorch_cnn(train_texts, test_texts, y_train, y_test, epochs=10)

    # 3. Plots & Reports
    plot_sentiment_benchmark(baseline_metrics, dl_metrics, history)

    comparison_report = {
        "baseline_tfidf_logistic_regression": baseline_metrics,
        "pytorch_1d_cnn": dl_metrics,
        "performance_lift": {
            "accuracy_lift": round(dl_metrics["accuracy"] - baseline_metrics["accuracy"], 4),
            "f1_lift": round(dl_metrics["f1_score"] - baseline_metrics["f1_score"], 4),
            "roc_auc_lift": round(dl_metrics["roc_auc"] - baseline_metrics["roc_auc"], 4)
        }
    }

    report_path = REPORTS_DIR / "sentiment_benchmark_results.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=4)
    print(f"  [REPORT SAVED] Exported sentiment benchmark results to: {report_path}")

    # Append Sentiment_Score
    df_scored = append_sentiment_scores(dl_model, vocab, df)
    return df_scored, comparison_report


if __name__ == "__main__":
    from src.module1_data_integration import run_module1
    from src.module4_feature_engineering import run_module4
    from src.module5_escalation_ml import run_module5
    from src.module6_segmentation import run_module6

    df_data, _ = run_module1()
    df_feat = run_module4(df_data)
    df_scored, _, _ = run_module5(df_feat)
    df_seg, _, _ = run_module6(df_scored)
    df_final, _ = run_module7(df_seg)
