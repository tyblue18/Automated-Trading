# Frontend Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Frontend

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### Single Text Analysis
- Paste any financial text (news article, tweet, etc.)
- Get instant UP/DOWN/NEUTRAL prediction
- Color-coded results for easy interpretation

### Batch Analysis
- Analyze multiple texts at once
- Upload CSV files or paste multiple texts
- View distribution charts and detailed results
- Download results as CSV

## Usage Examples

### Example 1: Analyze News Article
1. Copy a news article about a stock
2. Paste into the text area
3. Click "Analyze Sentiment"
4. See prediction (UP/DOWN/NEUTRAL)

### Example 2: Batch Analysis
1. Switch to "Batch Analysis" mode
2. Paste multiple texts (one per line) or upload CSV
3. Click "Analyze Batch"
4. View charts and download results

## Prediction Meanings

- **📈 UP**: Positive sentiment - Stock price predicted to increase
- **📉 DOWN**: Negative sentiment - Stock price predicted to decrease  
- **➡️ NEUTRAL**: Mixed or unclear sentiment - No clear direction

## Technical Details

The frontend uses:
- **Streamlit** for the web interface
- **Plotly** for interactive charts
- **Your ensemble_sentiment_analysis module** for predictions

## Troubleshooting

**First run is slow?**
- This is normal! Models load on first use (~3 seconds)
- Subsequent analyses are much faster (cached models)

**Error loading models?**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that `src/tfidf_lr_model.pkl` exists (optional, will work without it)

**Port already in use?**
- Streamlit uses port 8501 by default
- Change it: `streamlit run app.py --server.port 8502`

