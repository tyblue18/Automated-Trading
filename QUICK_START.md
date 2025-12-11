# Quick Start Guide

## Run the Frontend

```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## What You Can Do

### 1. Single Text Prediction
- Paste any financial news article or text
- Click "Analyze Sentiment"
- Get instant UP/DOWN/NEUTRAL prediction

**Example text to try:**
```
NVIDIA announced record-breaking quarterly earnings, with revenue up 200% year-over-year. 
The company's AI chips are in high demand as enterprises rush to adopt generative AI. 
Analysts predict continued strong growth in the coming quarters.
```

### 2. Batch Analysis
- Switch to "Batch Analysis" mode
- Paste multiple texts (one per line) or upload a CSV
- View charts showing sentiment distribution
- Download results as CSV

## Features

✅ **Real-time predictions** - Instant sentiment analysis  
✅ **Color-coded results** - Green (UP), Red (DOWN), Gray (NEUTRAL)  
✅ **Batch processing** - Analyze multiple texts at once  
✅ **Visualizations** - Pie charts and statistics  
✅ **CSV export** - Download results for further analysis  

## Troubleshooting

**First run is slow?**
- Normal! Models load on first use (~3 seconds)
- Subsequent analyses are instant (models cached)

**Port 8501 in use?**
```bash
streamlit run app.py --server.port 8502
```

**Import errors?**
```bash
pip install streamlit plotly
```

