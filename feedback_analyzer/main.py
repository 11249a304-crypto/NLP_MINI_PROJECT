"""
main.py  -  command-line version of the University Student Feedback Analyzer
===========================================================================
Runs the complete NLP project:
    dataset  ->  pre-processing demo  ->  training & evaluation  ->  insights
    ->  charts saved in outputs/  ->  (optional) interactive analyzer

Usage:
    python main.py                  # full run, then an interactive prompt
    python main.py --no-interactive # full run only
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw charts to files, no pop-up windows
import matplotlib.pyplot as plt

import nlp_pipeline as nlp
import visuals as vis

BASE = Path(__file__).parent
OUT = BASE / "outputs"

SAMPLE_COMMENTS = [
    "The professors are very supportive and the classes are really engaging.",
    "The syllabus is outdated and the lab computers don't work properly.",
    "Exams were not fair, the question paper was too lengthy!",
    "The library opens at nine and closes at eight.",
    "The office staff are not helpful at all.",
]

report_lines = []


def log(text=""):
    """Print to the screen AND keep a copy for outputs/evaluation_report.txt."""
    print(text)
    report_lines.append(str(text))


def section(title):
    log("\n" + "=" * 72)
    log(title)
    log("=" * 72)


def save(fig, name):
    if fig is None:
        return
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  saved  {path.relative_to(BASE)}")


def main():
    parser = argparse.ArgumentParser(description="University Student Feedback Analyzer")
    parser.add_argument("--data", default=str(BASE / "data" / "feedback_dataset.csv"),
                        help="CSV with columns: feedback, aspect, sentiment")
    parser.add_argument("--no-interactive", action="store_true",
                        help="skip the interactive prompt at the end")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- 1. data
    section("1. DATASET")
    df = nlp.load_dataset(args.data)
    log(f"Loaded {len(df)} feedback comments from {Path(args.data).name}")
    log("\nSentiment counts:")
    log(df["sentiment"].value_counts().to_string())
    log("\nAspect counts:")
    log(df["aspect"].value_counts().to_string())
    log("\nFirst 5 rows:")
    log(df[["feedback", "aspect", "sentiment"]].head().to_string(index=False))

    # ------------------------------------------------- 2. preprocessing demo
    section("2. NLP PRE-PROCESSING DEMO")
    for text in SAMPLE_COMMENTS[1:3]:
        steps = nlp.preprocess_steps(text)
        for key, value in steps.items():
            log(f"{key:24s}: {value}")
        log("-" * 72)

    # ------------------------------------------------------ 3. train & test
    section("3. TRAINING AND EVALUATION")
    log("Training models (80% train / 20% test) ...")
    bundle = nlp.train_all(df)

    log("\nModel comparison on the test set:")
    log(bundle["results"].round(3).to_string(index=False))
    log(f"\nBest model: {bundle['best_name']}")
    cv = bundle["cv_scores"]
    log(f"5-fold cross-validation F1 (best model): {cv.mean():.3f} +/- {cv.std():.3f}")
    log("\nClassification report - sentiment (best model, test set):")
    log(bundle["sentiment_report"])
    log(f"Aspect classifier accuracy (test set): {bundle['aspect_accuracy']:.3f}")
    log("\nClassification report - aspect:")
    log(bundle["aspect_report"])

    # ---------------------------------------------------------- 4. insights
    section("4. INSIGHTS FROM THE FEEDBACK")
    summary = nlp.aspect_summary(df)
    log("Satisfaction per aspect:")
    log(summary.to_string())

    phrases = nlp.key_phrases(df)
    log("\nTop complaint phrases (bigrams in negative comments):")
    for p, c in phrases["complaints"]:
        log(f"  {c:3d}  {p}")
    log("\nTop praise phrases (bigrams in positive comments):")
    for p, c in phrases["praises"]:
        log(f"  {c:3d}  {p}")

    adjectives = nlp.top_adjectives(df["feedback"], top_k=12)
    if adjectives:
        log("\nMost frequent opinion words (adjectives found with POS tagging):")
        log(", ".join(f"{w} ({c})" for w, c in adjectives))

    # ------------------------------------------------------------ 5. charts
    section("5. SAVING CHARTS")
    save(vis.plot_sentiment_distribution(df["sentiment"]), "01_sentiment_distribution.png")
    save(vis.plot_aspect_sentiment(df), "02_aspect_sentiment.png")
    save(vis.plot_satisfaction_index(summary), "03_satisfaction_index.png")
    save(vis.plot_model_comparison(bundle["results"]), "04_model_comparison.png")
    save(vis.plot_confusion_matrix(bundle["y_test"], bundle["test_preds"][bundle["best_name"]],
                                   nlp.SENTIMENT_LABELS,
                                   f"Sentiment - {bundle['best_name']}"), "05_confusion_matrix_sentiment.png")
    save(vis.plot_confusion_matrix(bundle["aspect_y_test"], bundle["aspect_pred"],
                                   bundle["aspect_labels"], "Aspect classifier"),
         "06_confusion_matrix_aspect.png")
    save(vis.plot_top_phrases(phrases["complaints"], "Top complaint phrases", "#E53935"),
         "07_top_complaints.png")
    save(vis.plot_top_phrases(phrases["praises"], "Top praise phrases", "#43A047"),
         "08_top_praises.png")
    save(vis.plot_wordcloud(bundle["df"]["clean"], "Most frequent words in all feedback"), "09_wordcloud.png")

    # ------------------------------------------------- 6. sample predictions
    section("6. PREDICTIONS ON NEW COMMENTS")
    for text in SAMPLE_COMMENTS:
        r = nlp.analyze_text(text, bundle)
        log(f'"{text}"')
        log(f"   -> sentiment: {r['sentiment']} ({r['sentiment_confidence']:.0%})   "
            f"aspect: {r['aspect']} ({r['aspect_confidence']:.0%})   "
            f"VADER: {r['vader_label']} ({r['vader_score']:+.2f})")

    (OUT / "evaluation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nFull report saved to {(OUT / 'evaluation_report.txt').relative_to(BASE)}")

    # ------------------------------------------------- 7. interactive prompt
    if not args.no_interactive:
        print("\n" + "=" * 72)
        print("INTERACTIVE ANALYZER  (type a comment and press Enter, or 'q' to quit)")
        print("=" * 72)
        while True:
            text = input("\nFeedback > ").strip()
            if text.lower() in {"q", "quit", "exit"}:
                break
            if not text:
                continue
            r = nlp.analyze_text(text, bundle)
            print(f"  Sentiment : {r['sentiment']} ({r['sentiment_confidence']:.0%} confidence)")
            print(f"  Aspect    : {r['aspect']} ({r['aspect_confidence']:.0%} confidence)")
            print(f"  VADER     : {r['vader_label']} (score {r['vader_score']:+.2f})")


if __name__ == "__main__":
    main()
