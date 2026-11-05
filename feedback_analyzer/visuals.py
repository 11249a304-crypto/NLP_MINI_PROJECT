"""
visuals.py
----------
All charts used by main.py (saved as PNG) and app.py (shown in Streamlit).
Every function returns a matplotlib Figure (or None if it cannot be drawn).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

COLORS = {"Positive": "#43A047", "Neutral": "#9E9E9E", "Negative": "#E53935"}
ORDER = ["Positive", "Neutral", "Negative"]


def plot_sentiment_distribution(sentiments, title="Sentiment distribution"):
    counts = pd.Series(list(sentiments)).value_counts().reindex(ORDER).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[c] for c in counts.index])
    ax.bar_label(bars)
    ax.set_title(title)
    ax.set_ylabel("Number of comments")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_aspect_sentiment(df, aspect_col="aspect", sentiment_col="sentiment",
                          title="Sentiment by aspect"):
    ct = pd.crosstab(df[aspect_col], df[sentiment_col]).reindex(columns=ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[COLORS[c] for c in ORDER], width=0.7)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Number of comments")
    ax.legend(title="Sentiment", loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig


def plot_satisfaction_index(summary):
    """Horizontal bar chart of the Satisfaction Index per aspect."""
    s = summary["Satisfaction Index"].sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.barh(s.index, s.values, color=[COLORS["Positive"] if v >= 0 else COLORS["Negative"] for v in s.values])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Satisfaction Index  =  (positive - negative) / total x 100")
    ax.set_title("Which aspects satisfy students most?")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    return fig


def plot_model_comparison(results):
    """Grouped bars (accuracy and F1) for every model."""
    data = results.set_index("Model")[["Accuracy", "F1-score"]]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    data.plot(kind="bar", ax=ax, color=["#1E88E5", "#FB8C00"], width=0.7)
    ax.set_ylim(0, 1.08)
    ax.set_title("Model comparison on the test set")
    ax.set_xlabel("")
    ax.spines[["top", "right"]].set_visible(False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    return fig


def plot_top_phrases(pairs, title, color="#1E88E5"):
    """Horizontal bar chart for a list of (phrase, count) pairs."""
    if not pairs:
        return None
    labels = [p for p, _ in pairs][::-1]
    values = [c for _, c in pairs][::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bars = ax.barh(labels, values, color=color)
    ax.bar_label(bars, padding=3)
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_wordcloud(texts, title="Most frequent words"):
    """Word cloud (needs the optional 'wordcloud' package; returns None if missing)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None
    text = " ".join(texts)
    if not text.strip():
        return None
    wc = WordCloud(width=900, height=420, background_color="white", colormap="viridis",
                   max_words=80).generate(text)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    return fig
