"""
Monitors RSS feeds for new articles that mention Nvidia in their title or
summary. Logs date, title, and body of relevant articles to CSV.

Expandable for multiple feeds and domains.
"""

import logging
import feedparser
import requests
from bs4 import BeautifulSoup
import csv
import time
from datetime import datetime
import os
from colorama import init, Fore, Style

from config import get_config
from ensemble_sentiment_analysis import analyze_sentiment
from google import genai

logger = logging.getLogger(__name__)

_cfg = get_config()
_news = _cfg["news"]
_tc = _cfg["target_company"]

FEED_URLS = _news["feed_urls"] + (["./testfeed.xml"] if _news.get("include_dev_feeds") else [])
CHECK_INTERVAL = _news["check_interval_seconds"]
_ALIASES = [a.lower() for a in _tc["aliases"]]

init(autoreset=True)
def print_colored_sentiment(sentiment):
    """Prints the text in color based on sentiment"""
    color_map = {
            "UP": Fore.GREEN + Style.BRIGHT,
            "DOWN": Fore.RED + Style.BRIGHT,
            "NEUTRAL": Fore.LIGHTBLACK_EX + Style.BRIGHT
    }

    print(f"[{color_map[sentiment]}{sentiment}{Style.RESET_ALL}]", end="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "..", _cfg["output"]["articles_csv"])

seen_links = set()

def get_article_text_generic(url):
    """Fetch text inside <p> tags"""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        body_text = " ".join(p.get_text() for p in paragraphs)
        return body_text.strip()
    except Exception as e:
        print(f"Error fetching article text: {e}")
        return ""

def get_article_text(url):
    """Decide which scraper to use based on domain"""

    # Example for later:
    # domain = urlparse(url).netloc
    # if "wired.com" in domain:
    #     return get_article_text_wired(url)
    # else:
    #     return get_article_text_generic(url)

    return get_article_text_generic(url)

def check_for_new_articles():
    """Parse RSS feeds and append new Nvidia articles to CSV"""
    global seen_links

    for feed_url in FEED_URLS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = getattr(entry, "summary", "").strip()

            # Get published date
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                # Standardize date for CSV
                published_csv = time.strftime("%Y-%m-%d", entry.published_parsed)

                # Pretty printing for console
                published_pretty = time.strftime("[%Y-%m-%d %a %H:%M]", entry.published_parsed)
            else:
                # If published date unavailable, use current time
                now = datetime.now()
                published_csv = now.strftime("%Y-%m-%d")
                published_pretty = now.strftime("[%Y-%m-%d %a %H:%M]")

            text_lower = (title + " " + summary).lower()
            if any(alias in text_lower for alias in _ALIASES):
                if link not in seen_links:
                    seen_links.add(link)

                    body = get_article_text(link)
                    sentiment = analyze_sentiment(body)
                    final_result = gemini_analysis(title, body, sentiment)

                    print(f"╠{published_pretty}", end="")
                    print_colored_sentiment(final_result)
                    print(f"[{Fore.BLUE}{Style.BRIGHT}\033]8;;{link}\033\\{title}\033]8;;\033\\{Style.RESET_ALL}]\n║")

                    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([published_csv, final_result, title, body])

def gemini_analysis(ARTICLE_TITLE, ARTICLE_BODY, SCORE): 
    client = genai.Client() 

    prompt = f"""You are a financial sentiment classifier. 
        You will be given an article title, an article body, and a suggested sentiment score. 
        The score is only a suggestion — you may override it if the text clearly supports a 
        different sentiment. 
        Your task: Output ONLY ONE WORD indicating the sentiment: 
        - "UP" for positive sentiment 
        - "DOWN" for negative sentiment 
        - "NEUTRAL" for mixed or unclear sentiment Rules: 
        - Output exactly one word and nothing else. No punctuation, no explanation. 
        - Ignore formatting, metadata, or irrelevant content in the article text. 

        Title: {ARTICLE_TITLE} 
        Body: {ARTICLE_BODY} 
        Suggested sentiment: {SCORE} 
        Respond with one word only: UP, DOWN, or NEUTRAL.""" 

    response = client.models.generate_content( 
        model="gemini-2.5-flash", 
        contents= prompt ) 

    result = response.text.strip()
    if result not in {"UP", "DOWN", "NEUTRAL"}:
        logger.warning("Gemini returned unexpected label %r; falling back to ensemble vote %r", result, SCORE)
        return SCORE
    return result


def main():
    while True:
        print(f"╔[{datetime.now().strftime('%Y-%m-%d %a %H:%M')}][Checking feeds for Nvidia articles]\n║")
        check_for_new_articles()
        print(f"╚[{datetime.now().strftime('%Y-%m-%d %a %H:%M')}][Checked all feeds]\n")

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()




