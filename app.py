"""
2-AMP Stock Sentiment Predictor - Streamlit Frontend
Simple web interface for predicting stock movement based on text sentiment.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os
import feedparser
import requests
from bs4 import BeautifulSoup
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ensemble_sentiment_analysis import analyze_sentiment, analyze_sentiment_batch

# Page configuration
st.set_page_config(
    page_title="2-AMP Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .prediction-up {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .prediction-down {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .prediction-neutral {
        background-color: #e2e3e5;
        color: #383d41;
        border: 2px solid #d6d8db;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📈 2-AMP Stock Sentiment Predictor</p>', unsafe_allow_html=True)
st.markdown("---")

# Helper functions for article fetching
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_rss_articles(feed_urls, target_companies, max_articles=20):
    """Fetch articles from RSS feeds."""
    articles = []
    seen_links = set()
    
    # Company aliases for better matching
    company_aliases = {
        "NVIDIA": ["nvidia", "nvda", "geforce", "cuda", "jensen huang"],
        "AMD": ["amd", "advanced micro devices", "ryzen", "radeon", "lisa su"],
        "TSMC": ["tsmc", "taiwan semiconductor", "tsm", "semiconductor"],
        "Apple": ["apple", "aapl", "iphone", "ipad", "macbook", "tim cook"],
        "Microsoft": ["microsoft", "msft", "azure", "windows", "satya nadella"],
        "Google": ["google", "googl", "goog", "alphabet", "sundar pichai"],
        "Amazon": ["amazon", "amzn", "aws", "jeff bezos", "alexa"],
        "Meta": ["meta", "facebook", "fb", "mark zuckerberg", "instagram"]
    }
    
    # Build search terms
    search_terms = []
    for company in target_companies:
        search_terms.append(company.lower())
        if company in company_aliases:
            search_terms.extend(company_aliases[company])
    
    for feed_url in feed_urls:
        try:
            # Add timeout and headers for better compatibility
            feed = feedparser.parse(feed_url)
            
            # Check if feed parsed successfully
            if feed.bozo:
                # Continue anyway, sometimes feeds have minor parsing issues but still work
                pass
            
            if not feed.entries:
                continue
            
            for entry in feed.entries[:max_articles * 3]:  # Check more entries to find matches
                title = getattr(entry, "title", "").strip()
                link = getattr(entry, "link", "").strip()
                summary = getattr(entry, "summary", "").strip()
                
                # Also try description if summary is empty
                if not summary:
                    summary = getattr(entry, "description", "").strip()
                
                if not title:
                    continue
                
                # Check if article mentions target companies (case-insensitive)
                text_lower = (title + " " + summary).lower()
                
                # Check against all search terms
                matches = [term for term in search_terms if term in text_lower]
                
                if matches and link not in seen_links:
                    seen_links.add(link)
                    
                    # Get published date
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        try:
                            published = time.strftime("%Y-%m-%d %H:%M", entry.published_parsed)
                        except:
                            published = datetime.now().strftime("%Y-%m-%d %H:%M")
                    elif hasattr(entry, "published"):
                        published = entry.published
                    else:
                        published = datetime.now().strftime("%Y-%m-%d %H:%M")
                    
                    # Extract domain name
                    try:
                        if "//" in link:
                            domain = link.split("//")[1].split("/")[0]
                        else:
                            domain = "Unknown"
                    except:
                        try:
                            domain = feed_url.split("/")[2] if "/" in feed_url else "Unknown"
                        except:
                            domain = "Unknown"
                    
                    articles.append({
                        "title": title,
                        "url": link,
                        "summary": summary[:300] + "..." if len(summary) > 300 else summary if summary else "No summary available",
                        "published": published,
                        "source": domain,
                        "matched_terms": ", ".join(set(matches[:3]))  # Show which terms matched
                    })
                    
                    if len(articles) >= max_articles:
                        break
            
        except Exception as e:
            # Silently continue on errors (feeds may be unavailable)
            continue
    
    return articles

def get_article_text(url):
    """Fetch full article text from URL."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        body_text = " ".join(p.get_text() for p in paragraphs)
        return body_text.strip()
    except Exception as e:
        return ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    st.subheader("Analysis Mode")
    analysis_mode = st.radio(
        "Choose analysis type:",
        ["Auto-Fetch Articles", "Single Text", "Batch Analysis"],
        help="Auto-Fetch automatically finds and analyzes recent articles. Single text analyzes one piece of text. Batch analyzes multiple texts at once."
    )
    
    # Auto-fetch settings
    if analysis_mode == "Auto-Fetch Articles":
        st.markdown("---")
        st.subheader("🔍 Search Settings")
        target_companies = st.multiselect(
            "Companies to search for:",
            ["NVIDIA", "AMD", "TSMC", "Apple", "Microsoft", "Google", "Amazon", "Meta"],
            default=["NVIDIA", "AMD", "TSMC"]
        )
        
        show_all_recent = st.checkbox("Show all recent articles (even if not matching)", value=True, 
                                     help="If no matching articles found, show recent financial news")
        
        max_articles = st.slider("Max articles to fetch:", 5, 50, 20)
        
        auto_refresh = st.checkbox("Auto-refresh every 5 minutes", value=False)
    
    st.markdown("---")
    st.subheader("📊 About")
    st.info("""
    **2-AMP** uses an ensemble of:
    - VADER Sentiment Analyzer
    - FinBERT (Financial BERT)
    - TF-IDF + Logistic Regression
    
    Predicts stock movement based on financial news sentiment.
    """)
    
    st.markdown("---")
    st.caption("💡 Tip: Use Auto-Fetch to automatically find and analyze recent news articles")

# Main content area
if analysis_mode == "Auto-Fetch Articles":
    st.header("🔍 Auto-Fetch & Analyze Articles")
    
    if not target_companies:
        st.warning("⚠️ Please select at least one company to search for.")
    else:
        # RSS feed URLs - Multiple sources for better coverage
        feed_urls = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AMD&region=US&lang=en-US",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=TSM&region=US&lang=en-US",
            "https://rss.cnn.com/rss/money_latest.rss",
            "https://feeds.reuters.com/reuters/businessNews",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # CNBC Business
        ]
        
        # Show feed status
        with st.expander("📡 RSS Feed Sources"):
            st.write("Fetching from:")
            for url in feed_urls:
                st.write(f"- {url}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            fetch_button = st.button("🔍 Fetch Articles", type="primary", use_container_width=True)
        
        if fetch_button or auto_refresh:
            with st.spinner(f"🔄 Fetching articles about {', '.join(target_companies) if target_companies else 'all companies'}..."):
                articles = fetch_rss_articles(feed_urls, target_companies, max_articles)
                
                # Fallback: fetch recent articles if no matches
                if not articles and show_all_recent:
                    st.info("🔄 No matching articles found. Fetching recent financial news...")
                    all_articles = []
                    seen = set()
                    
                    for feed_url in feed_urls[:4]:  # Try multiple feeds
                        try:
                            feed = feedparser.parse(feed_url)
                            if feed.entries:
                                for entry in feed.entries[:10]:
                                    title = getattr(entry, "title", "").strip()
                                    link = getattr(entry, "link", "").strip()
                                    
                                    if title and link and link not in seen:
                                        seen.add(link)
                                        summary = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
                                        
                                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                                            try:
                                                published = time.strftime("%Y-%m-%d %H:%M", entry.published_parsed)
                                            except:
                                                published = datetime.now().strftime("%Y-%m-%d %H:%M")
                                        else:
                                            published = datetime.now().strftime("%Y-%m-%d %H:%M")
                                        
                                        try:
                                            domain = link.split("//")[1].split("/")[0] if "//" in link else "Unknown"
                                        except:
                                            domain = "Recent News"
                                        
                                        all_articles.append({
                                            "title": title,
                                            "url": link,
                                            "summary": summary[:300] + "..." if len(summary) > 300 else summary or "No summary available",
                                            "published": published,
                                            "source": domain,
                                            "matched_terms": "Recent news"
                                        })
                                        
                                        if len(all_articles) >= max_articles:
                                            break
                        except Exception as e:
                            continue
                    
                    if all_articles:
                        articles = all_articles
                        st.success(f"✅ Found {len(articles)} recent articles")
                        st.info("💡 **Note:** These are recent financial articles. They may not specifically mention your selected companies.")
                    else:
                        st.error("❌ Could not fetch articles from RSS feeds.")
                        st.info("""
                        **Troubleshooting:**
                        1. Check your internet connection
                        2. Try using 'Single Text' mode instead
                        3. RSS feeds may be temporarily unavailable
                        """)
                
                if articles:
                    st.success(f"✅ Found {len(articles)} articles")
                    
                    # Analyze articles
                    with st.spinner("🔄 Analyzing sentiment for all articles..."):
                        article_texts = []
                        for article in articles:
                            # Try to get full text, fallback to summary
                            full_text = get_article_text(article["url"])
                            text_to_analyze = full_text if full_text else article["summary"]
                            article_texts.append(text_to_analyze)
                        
                        # Batch analyze
                        predictions = analyze_sentiment_batch(article_texts, batch_size=8)
                        
                        # Add predictions to articles
                        for i, article in enumerate(articles):
                            article["prediction"] = predictions[i] if i < len(predictions) else "NEUTRAL"
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Analysis Results")
                    
                    # Summary statistics
                    results_df = pd.DataFrame(articles)
                    prediction_counts = results_df['prediction'].value_counts()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Articles", len(results_df))
                    with col2:
                        up_count = (results_df['prediction'] == 'UP').sum()
                        st.metric("📈 UP", up_count, delta=f"{up_count/len(results_df)*100:.1f}%")
                    with col3:
                        down_count = (results_df['prediction'] == 'DOWN').sum()
                        st.metric("📉 DOWN", down_count, delta=f"{down_count/len(results_df)*100:.1f}%")
                    with col4:
                        neutral_count = (results_df['prediction'] == 'NEUTRAL').sum()
                        st.metric("➡️ NEUTRAL", neutral_count, delta=f"{neutral_count/len(results_df)*100:.1f}%")
                    
                    # Visualization
                    st.subheader("📊 Sentiment Distribution")
                    fig = px.pie(
                        values=prediction_counts.values,
                        names=prediction_counts.index,
                        color_discrete_map={'UP': '#28a745', 'DOWN': '#dc3545', 'NEUTRAL': '#6c757d'},
                        title="Article Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Articles list
                    st.subheader("📰 Articles & Predictions")
                    
                    # Filter by prediction
                    filter_pred = st.selectbox("Filter by prediction:", ["All", "UP", "DOWN", "NEUTRAL"])
                    filtered_df = results_df if filter_pred == "All" else results_df[results_df['prediction'] == filter_pred]
                    
                    # Display articles
                    for idx, article in filtered_df.iterrows():
                        # Color-coded container
                        if article['prediction'] == 'UP':
                            st.markdown("""
                            <div style='background-color: #d4edda; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid #28a745;'>
                            """, unsafe_allow_html=True)
                        elif article['prediction'] == 'DOWN':
                            st.markdown("""
                            <div style='background-color: #f8d7da; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid #dc3545;'>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='background-color: #e2e3e5; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border-left: 4px solid #6c757d;'>
                            """, unsafe_allow_html=True)
                        
                        col_pred, col_title = st.columns([1, 4])
                        with col_pred:
                            st.markdown(f"### {article['prediction']}")
                        with col_title:
                            st.markdown(f"**{article['title']}**")
                        
                        st.markdown(f"📅 {article['published']} | 🌐 {article['source']}")
                        st.markdown(f"💬 {article['summary']}")
                        st.markdown(f"[🔗 Read full article]({article['url']})")
                        st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"auto_fetched_articles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    if auto_refresh:
                        st.info("⏳ Auto-refresh enabled. Page will refresh in 5 minutes...")
                        time.sleep(300)
                        st.rerun()

elif analysis_mode == "Single Text":
    st.header("📝 Enter Text to Analyze")
    
    # Text input
    text_input = st.text_area(
        "Paste news article, tweet, or financial text here:",
        height=200,
        placeholder="Example: NVIDIA announced record-breaking quarterly earnings, with revenue up 200% year-over-year. The company's AI chips are in high demand..."
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    if analyze_button:
        if not text_input or not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔄 Analyzing sentiment... This may take a few seconds on first run (loading models)..."):
                try:
                    # Analyze sentiment
                    sentiment = analyze_sentiment(text_input)
                    
                    # Display prediction
                    st.markdown("---")
                    st.header("🎯 Prediction")
                    
                    # Color-coded prediction box
                    if sentiment == "UP":
                        st.markdown(
                            f'<div class="prediction-box prediction-up">📈 STOCK PRICE: {sentiment}</div>',
                            unsafe_allow_html=True
                        )
                        st.success("✅ **Positive sentiment detected** - Stock price is predicted to increase")
                    elif sentiment == "DOWN":
                        st.markdown(
                            f'<div class="prediction-box prediction-down">📉 STOCK PRICE: {sentiment}</div>',
                            unsafe_allow_html=True
                        )
                        st.error("❌ **Negative sentiment detected** - Stock price is predicted to decrease")
                    else:
                        st.markdown(
                            f'<div class="prediction-box prediction-neutral">➡️ STOCK PRICE: {sentiment}</div>',
                            unsafe_allow_html=True
                        )
                        st.info("⚪ **Neutral sentiment** - No clear direction predicted")
                    
                    # Additional info
                    with st.expander("ℹ️ How this prediction works"):
                        st.markdown("""
                        The system uses an **ensemble of three models**:
                        1. **VADER** - Lexicon-based sentiment analyzer
                        2. **FinBERT** - Financial domain-specific BERT model
                        3. **TF-IDF + LR** - Traditional ML classifier
                        
                        Each model votes on the sentiment, and the majority determines the final prediction.
                        If there's a tie, VADER breaks it (highest accuracy).
                        """)
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing text: {e}")
                    st.exception(e)

else:  # Batch Analysis
    st.header("📊 Batch Analysis")
    
    st.markdown("Analyze multiple texts at once (faster processing)")
    
    # Batch input options
    input_method = st.radio(
        "Input method:",
        ["Paste multiple texts (one per line)", "Upload CSV file"]
    )
    
    texts_to_analyze = []
    
    if input_method == "Paste multiple texts (one per line)":
        batch_text = st.text_area(
            "Enter multiple texts (one per line):",
            height=300,
            placeholder="NVIDIA stock surges after earnings beat\nAMD announces new GPU architecture\nTSMC reports strong quarterly results"
        )
        
        if batch_text:
            texts_to_analyze = [line.strip() for line in batch_text.split('\n') if line.strip()]
    
    else:  # CSV upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows from CSV")
                
                # Let user select text column
                if len(df.columns) > 0:
                    text_column = st.selectbox("Select column containing text:", df.columns)
                    if text_column:
                        texts_to_analyze = df[text_column].dropna().astype(str).tolist()
                        st.info(f"Found {len(texts_to_analyze)} texts to analyze")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    
    if st.button("🔍 Analyze Batch", type="primary"):
        if not texts_to_analyze:
            st.warning("⚠️ Please provide texts to analyze.")
        else:
            with st.spinner(f"🔄 Analyzing {len(texts_to_analyze)} texts... This may take a moment..."):
                try:
                    # Batch analyze
                    results = analyze_sentiment_batch(texts_to_analyze, batch_size=8)
                    
                    # Create results dataframe
                    results_df = pd.DataFrame({
                        'Text': texts_to_analyze,
                        'Prediction': results,
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    # Display results
                    st.markdown("---")
                    st.header("📊 Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Texts", len(results_df))
                    with col2:
                        up_count = (results_df['Prediction'] == 'UP').sum()
                        st.metric("📈 UP", up_count, delta=f"{up_count/len(results_df)*100:.1f}%")
                    with col3:
                        down_count = (results_df['Prediction'] == 'DOWN').sum()
                        st.metric("📉 DOWN", down_count, delta=f"{down_count/len(results_df)*100:.1f}%")
                    with col4:
                        neutral_count = (results_df['Prediction'] == 'NEUTRAL').sum()
                        st.metric("➡️ NEUTRAL", neutral_count, delta=f"{neutral_count/len(results_df)*100:.1f}%")
                    
                    # Visualization
                    st.subheader("📊 Prediction Distribution")
                    prediction_counts = results_df['Prediction'].value_counts()
                    
                    fig = px.pie(
                        values=prediction_counts.values,
                        names=prediction_counts.index,
                        color_discrete_map={'UP': '#28a745', 'DOWN': '#dc3545', 'NEUTRAL': '#6c757d'},
                        title="Sentiment Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name=f"sentiment_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing batch: {e}")
                    st.exception(e)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; padding: 1rem;'>
        <p>2-AMP: Automatic Media Processing - Algorithmic Market Predictor</p>
        <p>Built with Streamlit | Powered by VADER, FinBERT, and TF-IDF</p>
    </div>
    """,
    unsafe_allow_html=True
)

