# University Student Feedback Analyzer

An NLP mini project (CSE 10104). It reads student feedback comments and automatically finds
**how students feel** (Positive / Neutral / Negative) and **what they are talking about**
(Teaching, Course Content, Exams, Facilities, Administration). It then summarises which
areas satisfy students and which need improvement.

## 1. Setup

Python 3.9 or newer is required.

```
pip install -r requirements.txt
```

The first run downloads the small NLTK datasets automatically (internet needed once). If that
fails, run this once and try again:

```
python -m nltk.downloader stopwords wordnet omw-1.4 averaged_perceptron_tagger_eng averaged_perceptron_tagger vader_lexicon
```

`wordcloud` is optional. If it will not install, the project still works (a bar chart of the
most frequent words is shown instead of the word cloud).

## 2. How to run

| Command | What it does |
|---|---|
| `python main.py` | Full pipeline in the terminal: dataset summary, NLP steps, model training and evaluation, insights, saves all charts and a report into `outputs/`, then opens an interactive prompt. Add `--no-interactive` to skip the prompt. |
| `streamlit run app.py` | Opens the web interface in your browser (4 tabs: analyze a comment, dataset insights, model performance, batch CSV analysis). |
| `python generate_dataset.py` | Re-creates `data/feedback_dataset.csv` (already included). |

To test the batch tab, upload `data/sample_new_feedback.csv`.

## 3. Project structure

```
feedback_analyzer/
|-- data/
|   |-- feedback_dataset.csv        labelled dataset (feedback, aspect, sentiment)
|   `-- sample_new_feedback.csv     unlabelled comments to try the batch upload
|-- outputs/                        charts + evaluation_report.txt (created by main.py)
|-- generate_dataset.py             builds the dataset
|-- nlp_pipeline.py                 all NLP + machine-learning logic
|-- visuals.py                      all charts
|-- main.py                         command-line version
|-- app.py                          Streamlit web version
|-- requirements.txt
`-- README.md
```

## 4. NLP concepts used (and where)

| Concept | Where (`nlp_pipeline.py`) |
|---|---|
| Text cleaning, contraction expansion | `clean_text()` |
| Tokenization | `tokenize()` |
| Stop-word removal (negations such as "not" are kept on purpose) | `remove_stopwords()`, `KEEP_WORDS` |
| Stemming (Porter) | `stem_tokens()` |
| Lemmatization (WordNet) | `lemmatize_tokens()` |
| POS tagging, to find opinion words (adjectives) | `pos_tags()`, `top_adjectives()` |
| N-grams (frequent phrases) | `top_ngrams()`, `key_phrases()` |
| Bag of Words and TF-IDF (unigrams + bigrams) | `build_models()` |
| Naive Bayes and Logistic Regression classifiers | `build_models()`, `train_all()` |
| Rule-based sentiment baseline (VADER) | `vader_sentiment()` |
| Evaluation: accuracy, precision, recall, F1, confusion matrix, 5-fold cross-validation | `train_all()` |

## 5. How it works

1. **Pre-processing**: raw text -> lower-case -> expand contractions -> tokens -> remove stop words -> lemmas.
2. **Feature extraction**: the lemmas are converted to numbers with Bag of Words or TF-IDF (single words and word pairs).
3. **Classification**: three models are trained on 80% of the data and compared on the unseen 20%: Naive Bayes (Bag of Words), Naive Bayes (TF-IDF), Logistic Regression (TF-IDF). VADER is added as a no-training baseline. The best model (by macro F1) is re-trained on all data and used for predictions.
4. **Aspect detection**: a second TF-IDF + Logistic Regression model predicts the topic of the comment.
5. **Insights**: Satisfaction Index per aspect = (positive - negative) / total x 100, plus the most common complaint and praise phrases.

## 6. About the dataset

`data/feedback_dataset.csv` is a **synthetic demonstration dataset**: 60 hand-written comments plus
comments generated from sentence templates with some noise (typos, negations, informal text).
Because it is synthetic, accuracy numbers look better than they would on real student feedback.
To make the project stronger, collect 50-100 real comments (for example with a Google Form),
label them, and add them to the CSV with the same columns (`feedback`, `aspect`, `sentiment`).
No code changes are needed.

## 7. Limitations (good to mention in your report)

- A comment is assigned one sentiment and one aspect, so mixed comments such as "great teachers but a dirty canteen" are handled roughly.
- Sarcasm is not detected.
- The model only knows the vocabulary of its training data; very different wording lowers confidence.
- Results depend on the dataset size and quality.

## 8. Possible future work

Deep-learning models (BERT), multi-aspect detection per comment, support for regional languages, and a feedback dashboard for departments.
