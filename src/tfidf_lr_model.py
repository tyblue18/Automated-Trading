# tfidf_lr_model.py
#
# Implements a primitve base model. Classifies text as UP, DOWN, or NEUTRAl.
#
# Computs UP/DOWN/NEUTRAL labels using 3-day return.
#
# Matches daily stock data from 2011-2025 with NVDA forbes articles.

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging

from config import configure_cli_logging

logger = logging.getLogger(__name__)
configure_cli_logging()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTICLES_CSV = os.path.join(BASE_DIR, "../data/forbes_articles_738.csv")
STOCK_DATA_CSV = os.path.join(BASE_DIR, "../data/NVDA_yahoo_finance_data_2011_2025.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "tfidf_lr_model.pkl")

"""
Prepare Data
"""
articles = pd.read_csv(ARTICLES_CSV)
stock_data = pd.read_csv(STOCK_DATA_CSV)

# Parse article dates
articles["Time_clean"] = articles["Time"].str.rsplit(" ", n=1).str[0]
articles["Time_clean"] = pd.to_datetime(
    articles["Time_clean"], format="%b %d, %Y, %I:%M%p"
)
articles["Date"] = pd.to_datetime(articles["Time_clean"].dt.date)
articles = articles.sort_values("Date")

# Parse stock dates
stock_data["StockDate"] = pd.to_datetime(stock_data["Date"], format="%d-%b-%y")
stock_data = stock_data.sort_values("StockDate")

# Compute UP/DOWN/NEUTRAL labels using 3-day return
stock_data["Close_t"] = stock_data["Close"]
stock_data["Close_t3"] = stock_data["Close"].shift(-3)

# 3-day return: (price in 3 days - today's price) / today's price
stock_data["Return_3d"] = (stock_data["Close_t3"] - stock_data["Close_t"]) / stock_data["Close_t"]

# Drop rows where 3-day future data doesn't exist
stock_data = stock_data.dropna(subset=["Return_3d"])

# Calculated NVDA Volatility
# NEUTRAL if |return| < k × volatility
vol = stock_data["Return_3d"].std()

k = 0.35
UP_THRESHOLD = k * vol
DOWN_THRESHOLD = -k * vol

def classify_vol(r):
    if r > UP_THRESHOLD:
        return "UP"
    elif r < DOWN_THRESHOLD:
        return "DOWN"
    else:
        return "NEUTRAL"

stock_data["Label"] = stock_data["Return_3d"].apply(classify_vol)

stock_data = stock_data.drop(columns=["Close_t", "Close_t3"])

"""
Merge articles with stock data
"""
merged = pd.merge_asof(
    articles,
    stock_data[["StockDate", "Label"]],
    left_on="Date",
    right_on="StockDate",
    direction="forward"
)
merged = merged.dropna(subset=["Label"])

# Combine title + body into one text column
merged["text"] = merged["Title"].fillna("") + " " + merged["Body"].fillna("")

X = merged["text"].astype(str)
y = merged["Label"].astype(str)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english", 
    max_features=2000,
)
X_tfidf = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# Train Model
model = OneVsRestClassifier(LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
))
model.fit(X_train, y_train)

y_pred =  model.predict(X_test)

logger.info("Accuracy: %s", accuracy_score(y_test, y_pred))
logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))

# Save model
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump({
        "vectorizer": vectorizer,
        "model": model
    }, f)

logger.info("TF-IDF + Logistic Regression model saved to: %s", MODEL_SAVE_PATH)
