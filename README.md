# Detecting AI-generated text

A university seminar project that compares classical machine-learning pipelines for telling human-written essays from AI-generated ones.

## What it does

`drugi_domaci.py` trains and evaluates **20 pipelines**, every combination of:

| Step | Options |
| --- | --- |
| Preprocessing (NLTK) | lowercase, strip punctuation, remove English stop words; **with or without** WordNet lemmatization |
| Vectorizer | bag of words (`CountVectorizer`) or `TfidfVectorizer` |
| Classifier | Logistic Regression, Multinomial Naive Bayes, Decision Tree, Random Forest, Extra Trees |

Each pipeline is trained on 80 % of the data and scored by accuracy on the remaining 20 %. The script reports the best combination, then measures how accuracy changes with the size of the training set. The plots are in `Figure_*.png`, and the full write-up with results is in `Seminarski.pdf` (in Montenegrin).

## Data

`train_v2_drcat_02.csv` (about 97 MB) is the DAIGT v2 training set from Kaggle's "LLM – Detect AI Generated Text" competition. It has a `text` column and a `label` column (1 = AI-generated).

## Run

```sh
pip install pandas numpy matplotlib scikit-learn nltk
python drugi_domaci.py
```

Training all 20 pipelines, including the random forests, on the full dataset takes a while.

Note: the script turns off SSL certificate checks so NLTK can download its data on machines where that fails. If the download works on your machine, remove those two `ssl` lines.
