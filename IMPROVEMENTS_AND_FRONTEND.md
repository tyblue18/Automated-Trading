# Project Improvements & Frontend Development Plan

## Current Limitations & Areas for Improvement

### 1. Performance Optimizations

**Model Loading Inefficiency:**
- **Issue**: FinBERT model is loaded on every sentiment analysis call (line 41 in `ensemble_sentiment_analysis.py`)
- **Fix**: Implement model caching/singleton pattern
```python
# Cache FinBERT model globally
_finbert_model = None
_finbert_tokenizer = None

def get_finbert_model():
    global _finbert_model, _finbert_tokenizer
    if _finbert_model is None:
        _finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        _finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    return _finbert_model, _finbert_tokenizer
```

**TF-IDF Model Usage:**
- **Issue**: Code collects `base_vote` but doesn't use it in final voting (line 82)
- **Fix**: Either remove it or properly integrate it into ensemble

**Batch Processing:**
- **Issue**: Processes articles one-by-one
- **Fix**: Implement batch processing for sentiment analysis (especially FinBERT)

### 2. Architecture Improvements

**API Layer:**
- **Current**: Scripts run directly
- **Improvement**: Create REST API (Flask/FastAPI) for:
  - Real-time sentiment analysis
  - Historical data queries
  - Model predictions
  - Data collection triggers

**Database Integration:**
- **Current**: CSV files
- **Improvement**: Use PostgreSQL/TimescaleDB for:
  - Efficient querying
  - Time-series optimization
  - Better data relationships
  - Concurrent access

**Task Queue:**
- **Current**: Synchronous processing
- **Improvement**: Use Celery/Redis for:
  - Background data collection
  - Async sentiment analysis
  - Scheduled tasks (RSS polling)

**Configuration Management:**
- **Current**: Hardcoded values in scripts
- **Improvement**: Use config files (YAML/JSON) or environment variables

### 3. Feature Enhancements

**Sentiment Analysis:**
- Add confidence scores (probability distributions)
- Implement sentiment intensity (not just UP/DOWN/NEUTRAL)
- Add explainability (which words/phrases drove the sentiment)

**Data Collection:**
- Add more news sources (Bloomberg, Reuters, Financial Times)
- Social media integration (Twitter/X, Reddit)
- Earnings call transcripts
- Analyst reports

**Trading Signals:**
- Implement backtesting framework
- Add risk metrics (Sharpe ratio, max drawdown)
- Portfolio-level sentiment aggregation
- Multi-timeframe analysis (1d, 7d, 30d returns)

**Monitoring & Alerts:**
- Real-time dashboard
- Email/SMS alerts for significant sentiment shifts
- Performance tracking
- Error monitoring

### 4. Code Quality

**Error Handling:**
- More granular exception handling
- Retry logic improvements
- Better logging (structured logging)

**Testing:**
- Unit tests for core functions
- Integration tests for pipelines
- Model accuracy tests

**Documentation:**
- API documentation (if API added)
- Code comments
- Architecture diagrams
- Deployment guides

---

## Frontend Development Plan

### Frontend Architecture Options

#### Option 1: Web Dashboard (Recommended)
**Tech Stack**: React + TypeScript + Tailwind CSS

**Features:**
1. **Real-Time Monitoring Dashboard**
   - Live sentiment feed (WebSocket connection)
   - Company-specific views (NVDA, AMD, TSM)
   - Sentiment trends over time (charts)
   - Recent articles with sentiment labels

2. **Historical Analysis**
   - Date range selector
   - Sentiment distribution charts
   - Return vs. sentiment correlation
   - Model performance metrics

3. **Data Collection Management**
   - Trigger manual data collection
   - View collection status
   - Monitor API health
   - Data source configuration

4. **Trading Signals**
   - Current sentiment scores
   - Predicted price movements
   - Confidence intervals
   - Historical accuracy

5. **Model Management**
   - View ensemble predictions
   - Individual model outputs
   - Model comparison
   - Retrain/update models

**Key Components:**
```typescript
// Example component structure
- Dashboard (main view)
- SentimentChart (time series)
- ArticleFeed (real-time updates)
- ModelComparison (ensemble breakdown)
- Settings (configuration)
- Alerts (notifications)
```

**Backend API Endpoints Needed:**
```python
# FastAPI example
GET  /api/sentiment/latest          # Latest sentiment predictions
GET  /api/sentiment/historical      # Historical data
POST /api/sentiment/analyze         # Analyze custom text
GET  /api/articles/recent           # Recent articles
GET  /api/models/performance        # Model metrics
POST /api/collect/trigger           # Trigger data collection
GET  /api/health                    # System health
```

#### Option 2: Streamlit (Quick Prototype)
**Tech Stack**: Python Streamlit

**Pros:**
- Fast development (Python only)
- Built-in widgets and charts
- No separate frontend/backend needed
- Great for data visualization

**Cons:**
- Less customizable
- Not ideal for production
- Limited real-time capabilities

**Use Case**: Quick prototype or internal tool

#### Option 3: Next.js Full-Stack
**Tech Stack**: Next.js + TypeScript + Prisma + PostgreSQL

**Pros:**
- Modern, scalable architecture
- Server-side rendering
- API routes built-in
- Great for production

**Cons:**
- More complex setup
- Requires more development time

---

## Recommended Frontend Implementation

### Phase 1: Basic Dashboard (MVP)

**Tech Stack:**
- **Frontend**: React + TypeScript + Vite
- **UI Library**: Shadcn/ui + Tailwind CSS
- **Charts**: Recharts or Chart.js
- **Backend**: FastAPI (Python)
- **Real-time**: WebSockets (FastAPI WebSocket)

**Core Features:**
1. Real-time sentiment feed
2. Historical charts
3. Article browser
4. Basic model comparison

**File Structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── Dashboard.tsx
│   │   ├── SentimentChart.tsx
│   │   ├── ArticleFeed.tsx
│   │   └── ModelComparison.tsx
│   ├── api/
│   │   └── client.ts
│   ├── hooks/
│   │   └── useWebSocket.ts
│   └── App.tsx
backend/
├── api/
│   ├── routes/
│   │   ├── sentiment.py
│   │   ├── articles.py
│   │   └── models.py
│   └── main.py
├── services/
│   ├── sentiment_service.py
│   └── data_service.py
```

### Phase 2: Advanced Features

- User authentication
- Custom alerts/notifications
- Portfolio management
- Backtesting interface
- Model training interface

### Phase 3: Production Features

- Multi-user support
- Role-based access
- API rate limiting
- Caching layer (Redis)
- Monitoring & analytics

---

## Quick Start: Streamlit Prototype

For a quick prototype, here's a simple Streamlit app:

```python
# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from src.ensemble_sentiment_analysis import analyze_sentiment

st.set_page_config(page_title="2-AMP Dashboard", layout="wide")

st.title("2-AMP: Automated Market Predictor")

# Sidebar
st.sidebar.header("Controls")
ticker = st.sidebar.selectbox("Select Ticker", ["NVDA", "AMD", "TSM"])

# Load data
@st.cache_data
def load_data(ticker):
    df = pd.read_csv(f"data/{ticker}_edgar_labeled.csv")
    return df

df = load_data(ticker)

# Main dashboard
col1, col2, col3 = st.columns(3)
col1.metric("Total Articles", len(df))
col2.metric("Avg 3d Return", f"{df['ret_3d'].mean():.2%}")
col3.metric("Accuracy", "75%")  # Calculate from labels

# Charts
st.subheader("Sentiment Over Time")
fig = px.line(df, x='date', y='ret_3d', color='sentiment', 
              title='Returns by Sentiment')
st.plotly_chart(fig, use_container_width=True)

# Recent articles
st.subheader("Recent Articles")
st.dataframe(df[['date', 'title', 'sentiment', 'ret_3d']].head(20))

# Custom analysis
st.subheader("Analyze Custom Text")
text_input = st.text_area("Enter text to analyze")
if st.button("Analyze"):
    sentiment = analyze_sentiment(text_input)
    st.write(f"Sentiment: **{sentiment}**")
```

**Run with:**
```bash
streamlit run app.py
```

---

## Integration Architecture

```
┌─────────────────┐
│   Frontend      │
│   (React/Next)  │
└────────┬────────┘
         │ HTTP/WebSocket
┌────────▼────────┐
│   FastAPI       │
│   Backend       │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    │         │          │          │
┌───▼───┐ ┌──▼──┐  ┌────▼────┐ ┌──▼──┐
│ Redis │ │PostgreSQL│ │Celery│ │Models│
│ Cache │ │Database │ │Tasks │ │(ML) │
└───────┘ └─────────┘ └───────┘ └─────┘
```

---

## Next Steps

1. **Immediate**: Create FastAPI backend with basic endpoints
2. **Week 1**: Build React dashboard with real-time feed
3. **Week 2**: Add charts and historical analysis
4. **Week 3**: Implement WebSocket for live updates
5. **Week 4**: Add model comparison and settings

Would you like me to:
1. Create a FastAPI backend structure?
2. Build a React frontend starter?
3. Create a Streamlit prototype?
4. Implement specific improvements to the existing code?

