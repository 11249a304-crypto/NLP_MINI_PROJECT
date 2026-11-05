"""
app.py  -  Streamlit web interface for the University Student Feedback Analyzer

Run with:   streamlit run app.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import nlp_pipeline as nlp
import visuals as vis

st.set_page_config(page_title="University Student Feedback Analyzer",
                   page_icon="🎓", layout="wide")

DATA_PATH = Path(__file__).parent / "data" / "feedback_dataset.csv"


@st.cache_resource(show_spinner="Training the NLP models (takes a few seconds)...")
def get_bundle():
    return nlp.train_all(nlp.load_dataset(DATA_PATH))


def show(fig):
    """Display a matplotlib figure, then free its memory."""
    if fig is not None:
        st.pyplot(fig)
        plt.close(fig)


bundle = get_bundle()
df = bundle["df"]

st.title("🎓 University Student Feedback Analyzer")
st.caption("An NLP mini project: tokenization, lemmatization, POS tagging, n-grams, "
           "TF-IDF, Naive Bayes / Logistic Regression and VADER, applied to student feedback.")

tab_analyze, tab_insights, tab_models, tab_batch = st.tabs(
    ["🔍 Analyze a comment", "📊 Dataset insights", "🧪 Model performance", "📁 Batch analysis"])

# =====================================================================
# TAB 1 - analyse one comment
# =====================================================================
with tab_analyze:
    st.subheader("Analyze a single feedback comment")
    text = st.text_area(
        "Type or paste a student feedback comment:",
        value="The professors are very supportive and the classes are really engaging.",
        height=110)
    st.caption("Try: *The syllabus is outdated and the lab computers are slow.*  |  "
               "*The exam was not fair.*  |  *The library opens at 9 am.*")

    if st.button("Analyze feedback", type="primary"):
        if not text.strip():
            st.warning("Please enter some feedback first.")
        else:
            res = nlp.analyze_text(text, bundle)

            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment (ML model)", res["sentiment"])
            c1.caption(f"Confidence: {res['sentiment_confidence']:.0%}")
            c2.metric("Aspect", res["aspect"])
            c2.caption(f"Confidence: {res['aspect_confidence']:.0%}")
            c3.metric("VADER (rule-based)", res["vader_label"])
            c3.caption(f"Compound score: {res['vader_score']:+.2f}")

            st.markdown("**Probability of each sentiment class**")
            st.bar_chart(pd.Series(res["probs"]))

            with st.expander("See the NLP pre-processing steps for this comment", expanded=True):
                steps = nlp.preprocess_steps(text)
                rows = []
                for key, value in steps.items():
                    if key == "POS tags":
                        value = "  ".join(f"{w}/{t}" for w, t in value) or "(tagger data not available)"
                    elif isinstance(value, list):
                        value = ", ".join(value)
                    rows.append({"Step": key, "Output": value})
                st.dataframe(pd.DataFrame(rows), hide_index=True)

# =====================================================================
# TAB 2 - dataset insights
# =====================================================================
with tab_insights:
    st.subheader("What are students saying?")
    n_total = len(df)
    pct_pos = (df["sentiment"] == "Positive").mean() * 100
    pct_neg = (df["sentiment"] == "Negative").mean() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Comments analysed", n_total)
    m2.metric("Positive", f"{pct_pos:.0f}%")
    m3.metric("Negative", f"{pct_neg:.0f}%")
    m4.metric("Aspects covered", df["aspect"].nunique())

    left, right = st.columns(2)
    with left:
        show(vis.plot_sentiment_distribution(df["sentiment"]))
    with right:
        show(vis.plot_aspect_sentiment(df))

    summary = nlp.aspect_summary(df)
    left, right = st.columns(2)
    with left:
        show(vis.plot_satisfaction_index(summary))
    with right:
        st.markdown("**Satisfaction table**")
        st.dataframe(summary.reset_index(), hide_index=True)
        st.caption("Satisfaction Index = (positive - negative) / total x 100")

    phrases = nlp.key_phrases(df)
    left, right = st.columns(2)
    with left:
        show(vis.plot_top_phrases(phrases["complaints"], "Top complaint phrases (bigrams)", "#E53935"))
    with right:
        show(vis.plot_top_phrases(phrases["praises"], "Top praise phrases (bigrams)", "#43A047"))

    cloud = vis.plot_wordcloud(df["clean"], "Most frequent words in all feedback")
    if cloud is not None:
        show(cloud)
    else:
        show(vis.plot_top_phrases(nlp.top_ngrams(df["feedback"], n=1, top_k=15),
                                  "Most frequent words (install 'wordcloud' for a word cloud)"))

    adjectives = nlp.top_adjectives(df["feedback"], top_k=12)
    if adjectives:
        st.markdown("**Most common opinion words (adjectives found with POS tagging):** "
                    + ", ".join(f"{w} ({c})" for w, c in adjectives))

    with st.expander("View the dataset"):
        st.dataframe(df[["feedback", "aspect", "sentiment"]], hide_index=True)

# =====================================================================
# TAB 3 - model performance
# =====================================================================
with tab_models:
    st.subheader("How well do the models work?")
    st.write("The dataset is split 80% / 20% (train / test). Every model is trained on the "
             "training part and scored on comments it has never seen.")
    st.dataframe(bundle["results"].round(3), hide_index=True)
    show(vis.plot_model_comparison(bundle["results"]))

    cv = bundle["cv_scores"]
    st.success(f"Best model: **{bundle['best_name']}**  |  5-fold cross-validation F1 = "
               f"{cv.mean():.3f} +/- {cv.std():.3f}")

    left, right = st.columns(2)
    with left:
        show(vis.plot_confusion_matrix(bundle["y_test"], bundle["test_preds"][bundle["best_name"]],
                                       nlp.SENTIMENT_LABELS, "Sentiment classifier"))
        st.code(bundle["sentiment_report"])
    with right:
        show(vis.plot_confusion_matrix(bundle["aspect_y_test"], bundle["aspect_pred"],
                                       bundle["aspect_labels"], "Aspect classifier"))
        st.code(bundle["aspect_report"])

# =====================================================================
# TAB 4 - batch analysis
# =====================================================================
with tab_batch:
    st.subheader("Analyze many comments at once")
    st.write("Upload a CSV file that has a column named **feedback** (for example "
             "`data/sample_new_feedback.csv`). Each comment gets a sentiment and an aspect.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            batch = None

        if batch is not None:
            if "feedback" not in batch.columns:
                st.error("The CSV must contain a column named 'feedback'.")
            else:
                batch = batch.dropna(subset=["feedback"])
                result = nlp.analyze_dataframe(batch, bundle)
                st.dataframe(result, hide_index=True)

                left, right = st.columns(2)
                with left:
                    show(vis.plot_sentiment_distribution(result["predicted_sentiment"],
                                                         "Sentiment of uploaded comments"))
                with right:
                    show(vis.plot_aspect_sentiment(result, "predicted_aspect", "predicted_sentiment",
                                                   "Sentiment by aspect (uploaded comments)"))

                st.download_button("Download results as CSV",
                                   result.to_csv(index=False).encode("utf-8"),
                                   file_name="analysis_results.csv", mime="text/csv")
