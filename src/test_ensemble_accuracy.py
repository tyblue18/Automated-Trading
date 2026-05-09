# test_ensemble_accuracy.py
#
# Tests sentiment analysis ensemble using Kaggle dataset takala/financial_phrasebank

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

import logging

from config import configure_cli_logging
from ensemble_sentiment_analysis import analyze_sentiment, labels

logger = logging.getLogger(__name__)
configure_cli_logging()

df = pd.read_csv("../data/sentiment_analysis_for_financial_news.csv")

# Normalize column names
df.columns = [c.lower().strip() for c in df.columns]

# Convert dataset labels → UP/DOWN/NEUTRAL
mapping = {
    "positive": "UP",
    "negative": "DOWN",
    "neutral": "NEUTRAL"
}

df["mapped_label"] = df["sentiment"].map(mapping)

# Remove rows where label couldn’t be mapped
df = df.dropna(subset=["mapped_label"])

y_true = []
y_pred = []

logger.info("Evaluating %d samples...", len(df))

for _, row in tqdm(df.iterrows(), total=len(df)):
    text = row["phrase"]
    true_label = row["mapped_label"]

    pred_label = analyze_sentiment(text)

    y_true.append(true_label)
    y_pred.append(pred_label)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, labels=labels)
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Output
logger.info("Accuracy: %s", accuracy)
logger.info("Classification Report:\n%s", report)
logger.info("Confusion Matrix (rows=true, cols=pred):\n%s", pd.DataFrame(cm, index=labels, columns=labels))
