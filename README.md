# Fake News Detector

Classifies news text as real or fake with TF-IDF features and four scikit-learn models
(Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), and can pull and summarize an article from a URL.

## Files
- `analyzer.py`: loads the dataset, cleans text (lowercase; strips URLs, HTML, punctuation, digits), splits 70/30,
  trains the four classifiers, and prints predictions for a sample article via `manual_testing(text)`.
- `FakeNews.py`: prompts for an article URL, downloads it with `newspaper3k`, and prints the title and an NLP summary.
  (Wiring the summary into the classifier is left commented out.)

## Data
[Fake and Real News dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
Download `True.csv` and `Fake.csv` into the repo root (they're not committed).

## Run
```bash
pip install pandas numpy scikit-learn nltk textblob newspaper3k lxml_html_clean
python analyzer.py     # train + classify the sample article
python FakeNews.py     # summarize an article from a URL
```

## Stack
Python · pandas · scikit-learn · NLTK · newspaper3k
