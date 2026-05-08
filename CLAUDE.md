# CLAUDE.md — 2-AMP (Automated Market Predictor)

## Project Purpose

Financial sentiment analysis system that predicts stock price movement (UP / DOWN / NEUTRAL)
from news articles and SEC filings. The core pipeline collects text from RSS feeds, Google News,
Forbes, and the SEC EDGAR API; classifies sentiment using a three-model voting ensemble; and
optionally applies a Gemini LLM override before writing results to CSV or displaying them in a
Streamlit dashboard.

Primary tickers tracked: **NVDA, AMD, TSM**.

---

## Module Responsibilities

| File | Role |
|------|------|
| `app.py` | Streamlit frontend — three modes: Auto-Fetch Articles (RSS), Single Text, Batch Analysis |
| `src/ensemble_sentiment_analysis.py` | Core voting ensemble: VADER + FinBERT (`yiyanghkust/finbert-tone`) + TF-IDF/LR; VADER breaks ties; exposes `analyze_sentiment()` and `analyze_sentiment_batch()` |
| `src/news_sentiment_analysis.py` | Continuous RSS monitor — fetches Forbes feeds every 10 min, filters for Nvidia mentions, runs ensemble then Gemini 2.5 Flash override, appends results to `data/nvidia_articles.csv` |
| `src/pipeline_edgar.py` | One-shot data pipeline — downloads SEC filings for NVDA/AMD/TSM (2008–2018), fetches full text, computes 3-day/5-day price returns, writes labeled CSVs to `data/` |
| `src/article_scraper.py` | One-shot script — scrapes Forbes article text from URLs in `forbes_search.csv`, outputs `forbes_articles.csv` |
| `src/article_finder.py` | One-shot script — Google News search for Forbes NVIDIA articles by year, outputs `forbes_search.csv` |
| `src/tfidf_lr_model.py` | Trains the TF-IDF + Logistic Regression classifier and saves `src/tfidf_lr_model.pkl` |
| `src/test_ensemble_accuracy.py` | Evaluates ensemble accuracy against the FinancialPhraseBank dataset |
| `src/config.py` | Cached config loader — `get_config()` returns the parsed `config.yaml` dict (`lru_cache`, read once per process) |
| `src/merge_data.py` | Utility — merges multiple labeled CSV sources |
| `src/LLM_proc.py` | Experimental LLM processing helper (incomplete) |
| `config.yaml` | Central config: tickers, EDGAR dates, RSS feeds, VADER thresholds, model paths, logging |

---

## How to Run

### Streamlit Dashboard
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`. First run is slow — FinBERT (~500 MB) downloads from HuggingFace.

### RSS Monitor (requires Gemini API key)
```bash
# Set GOOGLE_API_KEY or GEMINI_API_KEY as required by google-genai
python src/news_sentiment_analysis.py
```
Polls Forbes RSS feeds every 10 minutes. Writes to `data/nvidia_articles.csv`.

### EDGAR Data Pipeline
```bash
python src/pipeline_edgar.py
```
Downloads SEC filings, fetches full text, labels rows with 3d/5d returns.
Writes `data/{TICKER}_edgar_labeled.csv` and `data/all_edgar_labeled.csv`.

### Data Collection Scripts (run once, in order)
```bash
# 1. Find article links
python src/article_finder.py        # writes forbes_search.csv

# 2. Scrape article text
python src/article_scraper.py       # reads forbes_search.csv, writes forbes_articles.csv
```
**Note:** Both scripts execute at module level (no `if __name__ == "__main__"` guard).
Do not import them — run them directly.

### Train TF-IDF model
```bash
python src/tfidf_lr_model.py
```
Writes `src/tfidf_lr_model.pkl`.

### Evaluate ensemble
```bash
python src/test_ensemble_accuracy.py
```

---

## Tests

No formal test suite. `src/test_ensemble_accuracy.py` is an integration-style accuracy check
against `data/sentiment_analysis_for_financial_news.csv` (FinancialPhraseBank).
No pytest, unittest, or CI configuration exists.

---

## Linter / Formatter

None configured. No `.flake8`, `.pylintrc`, or `pyproject.toml` exists.
If adding one, `flake8` or `ruff` are recommended starting points.

---

## Known Dead Code

### `finBERT/` subdirectory
The local `finBERT/` directory is an external repo dropped into the project and **is not used
anywhere**. The ensemble loads `yiyanghkust/finbert-tone` from HuggingFace at runtime via
`transformers.AutoModelForSequenceClassification`. The local submodule can be safely ignored
or deleted.

### Commented-out deduplication block in `src/article_finder.py` (lines 78–82)
Dead code left from an earlier refactor.

### `data/nvidia_articles.csv`
Empty placeholder file. Only populated when `src/news_sentiment_analysis.py` has run.

### `README.md`
Empty file (single newline).

---

## Coding Conventions

**Use `logging`, not `print()`.**
`ensemble_sentiment_analysis.py` and `pipeline_edgar.py` already do this correctly.
`news_sentiment_analysis.py`, `article_scraper.py`, and `article_finder.py` still use `print()`.
New code should use the module-level logger pattern:
```python
import logging
logger = logging.getLogger(__name__)
logger.info("...")
```

**Use type hints (PEP 484).**
`ensemble_sentiment_analysis.py` and `pipeline_edgar.py` already have type hints.
New functions should annotate parameters and return types.

**No bare `except` clauses.**
Use `except Exception as e:` at minimum. Bare `except:` catches `KeyboardInterrupt` and
`SystemExit`. Current violations:
- `src/ensemble_sentiment_analysis.py:288` — bare `except:` inside `analyze_sentiment_batch`

**Read config.yaml instead of hardcoding constants.**
`pipeline_edgar.py` duplicates many values already in `config.yaml` as module-level constants.
New pipeline code should load settings via `src/config_loader.py`.

**One-shot scripts belong outside `src/`.**
`article_finder.py` and `article_scraper.py` are not importable modules — they execute at
module level. They should either be moved to a top-level `scripts/` directory or wrapped in
`if __name__ == "__main__":`.

---

## Notable Architecture Notes

**Ensemble voting logic (ensemble_sentiment_analysis.py):**
Three models vote; majority wins. If all three disagree (three-way tie), VADER decides.
If `tfidf_lr_model.pkl` is missing, TF-IDF returns `None` and is excluded from the vote.

**Gemini override in news_sentiment_analysis.py:**
The RSS monitor pipeline is: ensemble vote → `gemini_analysis()` → final label.
The Gemini call can override the ensemble result. This fourth model is **not present** in the
Streamlit frontend, which only calls `analyze_sentiment()` / `analyze_sentiment_batch()`.

**SEC user-agent string:**
`pipeline_edgar.py` line 40 hardcodes `"contact: you@example.com"` in the `User-Agent` header
sent to SEC APIs. This should be updated to a real contact address before production use.

**`config.yaml` `target_company`:**
`news_sentiment_analysis.py` reads `target_company.aliases` for its RSS filter. The Streamlit
frontend has its own independent company list and alias map hardcoded in `app.py` (lines 86–95).
These two are still not shared.

---

## Known issues / future work

**Labeling scheme in `pipeline_edgar.py`:**
The `label_with_returns()` function uses a binary `ret > 0` split — every filing is labeled
either UP or DOWN, no NEUTRAL class. The original audit referenced a `k=0.35` volatility
threshold that does not exist in the code. Worth revisiting the labeling scheme separately —
three-class labels (UP / DOWN / NEUTRAL) with a volatility-aware threshold would likely
improve training signal, but that is a modeling decision, not a config swap.
