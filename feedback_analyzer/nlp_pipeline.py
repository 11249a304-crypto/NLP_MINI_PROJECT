"""
nlp_pipeline.py
===============
Core NLP logic for the University Student Feedback Analyzer.

NLP concepts used (in the order they appear in the pipeline)
------------------------------------------------------------
 1. Text cleaning            - lower-casing, URL removal, contraction expansion
 2. Tokenization             - splitting text into words (NLTK RegexpTokenizer)
 3. Stop-word removal        - dropping filler words, but KEEPING negations ("not")
 4. Stemming                 - Porter stemmer (shown for comparison)
 5. Lemmatization            - WordNet lemmatizer (used for the model)
 6. POS tagging              - to find the opinion words (adjectives)
 7. N-grams                  - frequent phrases such as "poor lab"
 8. Feature extraction       - Bag of Words and TF-IDF (with unigrams + bigrams)
 9. Classification           - Naive Bayes and Logistic Regression
10. Rule-based sentiment     - VADER lexicon, used as a baseline
11. Evaluation               - accuracy, precision, recall, F1, confusion matrix, cross-validation
"""
import re
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.util import ngrams

from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# --------------------------------------------------------------------------
# NLTK data (downloaded automatically the first time)
# --------------------------------------------------------------------------
# (resource paths - any one is enough,  package names to download if missing)
_NLTK_NEEDS = [
    (["corpora/stopwords"], ["stopwords"]),
    (["corpora/wordnet"], ["wordnet", "omw-1.4"]),
    (["taggers/averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger"],
     ["averaged_perceptron_tagger_eng", "averaged_perceptron_tagger"]),
    (["sentiment/vader_lexicon.zip"], ["vader_lexicon"]),
]


def _has_resource(path):
    for p in (path, path + ".zip"):
        try:
            nltk.data.find(p)
            return True
        except LookupError:
            pass
    return False


def ensure_nltk_data():
    """Download the NLTK corpora / models we need (only if they are missing)."""
    for paths, packages in _NLTK_NEEDS:
        if not any(_has_resource(p) for p in paths):
            for pkg in packages:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass


ensure_nltk_data()

_MISSING_DATA_HELP = (
    "NLTK data is missing and could not be downloaded automatically.\n"
    "Connect to the internet and run:\n"
    "  python -m nltk.downloader stopwords wordnet omw-1.4 "
    "averaged_perceptron_tagger_eng averaged_perceptron_tagger vader_lexicon"
)

# --------------------------------------------------------------------------
# 1-5. Pre-processing
# --------------------------------------------------------------------------
# Words that NLTK treats as stop words but that matter for opinions.
# "not good" must not become "good"!
KEEP_WORDS = {"no", "nor", "not", "never", "too", "very", "few", "more", "less",
              "most", "but", "against", "only"}

_tokenizer = RegexpTokenizer(r"[a-z]+")
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


@lru_cache(maxsize=1)
def get_stopwords():
    from nltk.corpus import stopwords
    try:
        return frozenset(stopwords.words("english")) - KEEP_WORDS
    except LookupError as exc:
        raise RuntimeError(_MISSING_DATA_HELP) from exc


def clean_text(text):
    """Lower-case, remove URLs, expand contractions ("don't" -> "do not")."""
    text = str(text).lower().replace("\u2019", "'")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'m\b", " am", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'d\b", " would", text)
    text = re.sub(r"'s\b", "", text)
    return text


def tokenize(text):
    """Split cleaned text into word tokens (letters only)."""
    return _tokenizer.tokenize(clean_text(text))


def remove_stopwords(tokens):
    stop = get_stopwords()
    return [t for t in tokens if t not in stop and len(t) > 1]


def stem_tokens(tokens):
    return [_stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens):
    """Noun pass then verb pass: 'classes' -> 'class', 'teaching' -> 'teach'."""
    try:
        return [_lemmatizer.lemmatize(_lemmatizer.lemmatize(t, pos="n"), pos="v") for t in tokens]
    except LookupError as exc:
        raise RuntimeError(_MISSING_DATA_HELP) from exc


def preprocess(text):
    """Full pipeline used for the ML models: clean -> tokenize -> stop words -> lemmas."""
    return " ".join(lemmatize_tokens(remove_stopwords(tokenize(text))))


def pos_tags(text):
    """Part-of-speech tags for the tokens of a sentence, e.g. [('lectures','NNS'), ('boring','JJ')]."""
    try:
        return nltk.pos_tag(tokenize(text))
    except LookupError:
        return []


def preprocess_steps(text):
    """Return every intermediate result - handy for demos, screenshots and the report."""
    tokens = tokenize(text)
    no_stop = remove_stopwords(tokens)
    return {
        "Original text": str(text),
        "Cleaned text": clean_text(text),
        "Tokens": tokens,
        "After stop-word removal": no_stop,
        "Stemmed (Porter)": stem_tokens(no_stop),
        "Lemmatized": lemmatize_tokens(no_stop),
        "POS tags": pos_tags(text),
    }


# --------------------------------------------------------------------------
# 6-7. POS-based opinion words and N-grams
# --------------------------------------------------------------------------
def top_adjectives(texts, top_k=15):
    """Most frequent adjectives (opinion words) found with POS tagging."""
    counter = Counter()
    stop = get_stopwords()
    for text in texts:
        for word, tag in pos_tags(text):
            if tag.startswith("JJ") and word not in stop and len(word) > 2:
                counter[word] += 1
    return counter.most_common(top_k)


def top_ngrams(texts, n=2, top_k=10):
    """Most frequent n-grams (phrases) after pre-processing."""
    counter = Counter()
    for text in texts:
        tokens = preprocess(text).split()
        for gram in ngrams(tokens, n):
            counter[" ".join(gram)] += 1
    return counter.most_common(top_k)


# --------------------------------------------------------------------------
# 10. Rule-based baseline: VADER
# --------------------------------------------------------------------------
_vader = None


def vader_sentiment(text):
    """Return (label, compound_score) from the VADER lexicon."""
    global _vader
    if _vader is None:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    score = _vader.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive", score
    if score <= -0.05:
        return "Negative", score
    return "Neutral", score


# --------------------------------------------------------------------------
# 8-9. Models
# --------------------------------------------------------------------------
def build_models():
    """Three feature-extraction + classifier combinations to compare."""
    return {
        "Naive Bayes (Bag of Words)": Pipeline([
            ("vec", CountVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB()),
        ]),
        "Naive Bayes (TF-IDF)": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.5)),
        ]),
        "Logistic Regression (TF-IDF)": Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=5)),
        ]),
    }


def load_dataset(path):
    df = pd.read_csv(path)
    needed = {"feedback", "sentiment", "aspect"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Dataset must contain the columns {sorted(needed)}")
    df = df.dropna(subset=["feedback", "sentiment", "aspect"]).reset_index(drop=True)
    df["sentiment"] = df["sentiment"].str.strip().str.capitalize()
    return df


# --------------------------------------------------------------------------
# 11. Training + evaluation
# --------------------------------------------------------------------------
def train_all(df, test_size=0.2, seed=42):
    """
    Train and evaluate every model. Returns a dictionary ("bundle") holding the
    trained models and all evaluation results, used by main.py and app.py.
    """
    df = df.copy()
    df["clean"] = df["feedback"].apply(preprocess)

    # ---------- sentiment ----------
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["sentiment"], test_size=test_size,
        stratify=df["sentiment"], random_state=seed)

    rows, test_preds, fitted = [], {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        p, r, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                     "Precision": p, "Recall": r, "F1-score": f1})
        fitted[name] = model
        test_preds[name] = pred

    best_name = max(rows, key=lambda r: r["F1-score"])["Model"]

    # VADER baseline (works on the raw text, needs no training)
    try:
        raw_test = df.loc[X_test.index, "feedback"]
        vader_pred = np.array([vader_sentiment(t)[0] for t in raw_test])
        p, r, f1, _ = precision_recall_fscore_support(y_test, vader_pred, average="macro", zero_division=0)
        rows.append({"Model": "VADER (rule-based, no training)",
                     "Accuracy": accuracy_score(y_test, vader_pred),
                     "Precision": p, "Recall": r, "F1-score": f1})
        test_preds["VADER (rule-based, no training)"] = vader_pred
    except (LookupError, ImportError):
        pass

    results = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)

    # 5-fold cross-validation of the best model (more reliable than one split)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(clone(build_models()[best_name]), df["clean"], df["sentiment"],
                                cv=skf, scoring="f1_macro")

    # final model = best architecture re-trained on ALL the data
    sentiment_model = clone(build_models()[best_name]).fit(df["clean"], df["sentiment"])

    # ---------- aspect ----------
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        df["clean"], df["aspect"], test_size=test_size, stratify=df["aspect"], random_state=seed)
    aspect_eval = build_models()["Logistic Regression (TF-IDF)"].fit(Xa_train, ya_train)
    aspect_pred = aspect_eval.predict(Xa_test)
    aspect_model = build_models()["Logistic Regression (TF-IDF)"].fit(df["clean"], df["aspect"])

    return {
        "df": df,
        "results": results,
        "best_name": best_name,
        "cv_scores": cv_scores,
        "y_test": np.array(y_test),
        "test_preds": test_preds,
        "sentiment_report": classification_report(
            y_test, test_preds[best_name], labels=SENTIMENT_LABELS, zero_division=0),
        "sentiment_model": sentiment_model,
        "aspect_labels": sorted(df["aspect"].unique()),
        "aspect_y_test": np.array(ya_test),
        "aspect_pred": aspect_pred,
        "aspect_accuracy": accuracy_score(ya_test, aspect_pred),
        "aspect_report": classification_report(ya_test, aspect_pred, zero_division=0),
        "aspect_model": aspect_model,
    }


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def analyze_text(text, bundle):
    """Analyse ONE comment: sentiment, aspect, confidences and the VADER baseline."""
    clean = preprocess(text)
    s_model, a_model = bundle["sentiment_model"], bundle["aspect_model"]

    s_proba = s_model.predict_proba([clean])[0]
    a_proba = a_model.predict_proba([clean])[0]
    s_idx, a_idx = int(np.argmax(s_proba)), int(np.argmax(a_proba))

    try:
        v_label, v_score = vader_sentiment(text)
    except (LookupError, ImportError):
        v_label, v_score = "n/a", 0.0

    return {
        "sentiment": s_model.classes_[s_idx],
        "sentiment_confidence": float(s_proba[s_idx]),
        "aspect": a_model.classes_[a_idx],
        "aspect_confidence": float(a_proba[a_idx]),
        "vader_label": v_label,
        "vader_score": float(v_score),
        "probs": {c: float(p) for c, p in zip(s_model.classes_, s_proba)},
    }


def analyze_dataframe(df, bundle, text_col="feedback"):
    """Analyse many comments at once (used for the CSV upload)."""
    out = df.copy().reset_index(drop=True)
    clean = out[text_col].astype(str).apply(preprocess)

    s_proba = bundle["sentiment_model"].predict_proba(clean)
    a_proba = bundle["aspect_model"].predict_proba(clean)
    out["predicted_sentiment"] = bundle["sentiment_model"].classes_[s_proba.argmax(axis=1)]
    out["sentiment_confidence"] = s_proba.max(axis=1).round(3)
    out["predicted_aspect"] = bundle["aspect_model"].classes_[a_proba.argmax(axis=1)]

    try:
        out["vader_sentiment"] = [vader_sentiment(t)[0] for t in out[text_col].astype(str)]
    except (LookupError, ImportError):
        pass
    return out


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------
def aspect_summary(df, aspect_col="aspect", sentiment_col="sentiment"):
    """Counts per aspect and a Satisfaction Index = (positive - negative) / total x 100."""
    ct = pd.crosstab(df[aspect_col], df[sentiment_col])
    ct = ct.reindex(columns=["Positive", "Neutral", "Negative"], fill_value=0)
    ct["Total"] = ct.sum(axis=1)
    ct["Satisfaction Index"] = ((ct["Positive"] - ct["Negative"]) / ct["Total"] * 100).round(1)
    return ct.sort_values("Satisfaction Index", ascending=False)


def key_phrases(df, sentiment_col="sentiment", text_col="feedback", n=2, top_k=8):
    """Most common phrases in negative comments (complaints) and positive comments (praise)."""
    neg = df.loc[df[sentiment_col] == "Negative", text_col]
    pos = df.loc[df[sentiment_col] == "Positive", text_col]
    return {"complaints": top_ngrams(neg, n=n, top_k=top_k),
            "praises": top_ngrams(pos, n=n, top_k=top_k)}
