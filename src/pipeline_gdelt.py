# src/pipeline_gdelt.py
import logging
import os, io, re, time, requests
import pandas as pd
import yfinance as yf
from datetime import timedelta
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

OUT_DIR = "data"; os.makedirs(OUT_DIR, exist_ok=True)

COMPANY_ALIASES = {
    "NVDA": ["NVIDIA","NVDA","GeForce","CUDA","H100","A100","GPU","graphics card","graphics processor","Jensen Huang","Kepler","Maxwell","Pascal","Volta","Tegra"],
    "AMD":  ["AMD","Advanced Micro Devices","Ryzen","EPYC","Radeon","GPU","CPU","graphics card","Lisa Su","Bulldozer"],
    "TSM":  ["TSMC","Taiwan Semiconductor","TSM","semiconductor","chip fabrication","foundry","wafer","node","28nm","14nm","7nm"],
}
SECTOR_TERMS = ["semiconductor","chip","GPU","foundry","graphics","datacenter","AI chip","machine learning"]

NEWS_START = "2010-01-01"; NEWS_END = "2016-12-31"
PRICE_START = "2009-12-01"; PRICE_END = "2017-01-31"
HORIZONS = [3,5]
RUN_FULL = False  # first run does a short test

def day_range(start_date: str, end_date: str):
    cur = pd.to_datetime(start_date).normalize()
    end = pd.to_datetime(end_date).normalize()
    while cur <= end:
        s = cur.strftime("%Y%m%d")+"000000"; e = cur.strftime("%Y%m%d")+"235959"
        yield s, e; cur += pd.Timedelta(days=1)

def month_range(start_date: str, end_date: str):
    start = pd.to_datetime(start_date).replace(day=1)
    end = (pd.to_datetime(end_date)+pd.offsets.MonthEnd(0)).normalize()
    cur = start
    while cur <= end:
        s = cur.strftime("%Y%m%d")+"000000"
        e = (cur+pd.offsets.MonthEnd(0)).strftime("%Y%m%d")+"235959"
        yield s, e; cur = (cur+pd.offsets.MonthBegin(1)).normalize()

def _normalize_news_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    col = {c.lower(): c for c in df.columns}
    def pick(*names): 
        for n in names:
            if n in col: return df[col[n]]
        return pd.Series([None]*len(df))
    if "seendate" in col:
        dt = pd.to_datetime(df[col["seendate"]].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    elif "date" in col:
        dt = pd.to_datetime(df[col["date"]].astype(str).str[:8], format="%Y%m%d", errors="coerce")
    else:
        dt = pd.NaT
    out = pd.DataFrame({
        "ticker": ticker,
        "date": dt,
        "title": pick("title"),
        "url": pick("url","documentidentifier"),
        "domain": pick("domain","sourcecommonname"),
        "language": pick("language","sourcelanguage"),
        "content": pick("content"),
    })
    out = out[~out["date"].isna()].reset_index(drop=True)
    for c in ["title","url","domain","language","content"]:
        out[c] = out[c].astype(str).replace({"None":""}).str.strip()
    return out

def gdelt_query(query, start_dt, end_dt, maxrecords=250, sleep_sec=1.0, retries=4, timeout=30):
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    q = quote_plus(query)
    json_url = f"{base}?query={q}&mode=ArtList&format=json&sort=DateAsc&startdatetime={start_dt}&enddatetime={end_dt}&maxrecords={maxrecords}"
    csv_url  = f"{base}?query={q}&mode=ArtList&format=csv&sort=DateAsc&startdatetime={start_dt}&enddatetime={end_dt}&maxrecords={maxrecords}"
    headers = {"User-Agent":"stock-news-llm/0.1 you@example.com"}
    delay = sleep_sec
    for _ in range(retries):
        try:
            r = requests.get(json_url, headers=headers, timeout=timeout)
            if r.status_code==200 and r.headers.get("Content-Type","").lower().startswith("application/json"):
                data = r.json(); time.sleep(delay); return pd.DataFrame(data.get("articles",[]))
            r2 = requests.get(csv_url, headers=headers, timeout=timeout)
            if r2.status_code==200 and r2.text.strip() and not r2.text.lstrip().startswith("<"):
                df = pd.read_csv(io.StringIO(r2.text), on_bad_lines="skip"); time.sleep(delay); return df
            time.sleep(delay); delay *= 2
        except Exception:
            time.sleep(delay); delay *= 2
    return pd.DataFrame()

def _build_query_for_ticker(ticker: str) -> str:
    terms = list({*COMPANY_ALIASES.get(ticker,[ticker]), *SECTOR_TERMS})
    return "("+ " OR ".join(terms) +")"

def _alias_regex(ticker:str)->re.Pattern:
    pats = [r"\b"+re.escape(a).replace(r"\ ", r"\s+")+r"\b" for a in COMPANY_ALIASES.get(ticker,[ticker])]
    return re.compile("|".join(pats), re.IGNORECASE)

def filter_hits_to_ticker(df, ticker):
    if df.empty: return df
    rgx = _alias_regex(ticker)
    mask = df["title"].fillna("").str.contains(rgx) | df["content"].fillna("").str_contains(rgx)
    # str.contains above; pandas <2.2 has no str_contains method; fallback:
    if "str_contains" in dir(pd.Series.str):
        pass
    else:
        mask = df["title"].fillna("").str.contains(rgx) | df["content"].fillna("").str.contains(rgx)
    return df[mask].reset_index(drop=True)

def fetch_news_for_ticker(ticker, start, end):
    # use daily slices for <=2014, monthly for later
    def windows():
        if pd.to_datetime(start).year <= 2014:
            return day_range(start, end)
        return month_range(start, end)
    frames = []
    query = _build_query_for_ticker(ticker)
    for s,e in windows():
        raw = gdelt_query(query, s, e)
        if not raw.empty: frames.append(_normalize_news_df(raw, ticker))
    if not frames: return pd.DataFrame()
    news = pd.concat(frames, ignore_index=True)
    news = filter_hits_to_ticker(news, ticker)
    news["title_lc"]=news["title"].str.lower()
    news = news.drop_duplicates(subset=["ticker","date","title_lc","url"]).drop(columns=["title_lc"]).reset_index(drop=True)
    return news

def fetch_prices(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty: return pd.DataFrame()
    df = df.reset_index().rename(columns={"Date":"date"})
    df["date"]=pd.to_datetime(df["date"])
    return df[["date","Open","High","Low","Close","Volume"]]

def first_on_or_after(prices, dt):
    sub = prices[prices["date"]>=dt]; return None if sub.empty else sub.iloc[0]

def label_with_returns(news, prices, horizons=(3,5)):
    if news.empty or prices.empty: return pd.DataFrame()
    rows=[]
    for _,r in news.iterrows():
        d0=r["date"]; row0 = first_on_or_after(prices, d0)
        if row0 is None: continue
        p0=float(row0["Close"])
        rec={"ticker":r["ticker"],"date":d0,"title":r.get("title","") or "","url":r.get("url","") or "","snippet":(r.get("content","") or "").strip(),"price_t0":p0}
        for h in horizons:
            rowh = first_on_or_after(prices, d0+pd.Timedelta(days=h))
            if rowh is None:
                rec[f"ret_{h}d"]=None; rec[f"label_{h}d"]=None
            else:
                ret=float(rowh["Close"])/p0-1.0; rec[f"ret_{h}d"]=ret; rec[f"label_{h}d"]="UP" if ret>0 else "DOWN"
        rows.append(rec)
    out=pd.DataFrame(rows).drop_duplicates(subset=["ticker","date","title","url"]).reset_index(drop=True)
    out["title"]=out["title"].fillna("").str.strip()
    out["snippet"]=out["snippet"].fillna("").str.replace(r"\s+"," ",regex=True).str.strip()
    return out

if __name__=="__main__":
    from config import configure_cli_logging
    configure_cli_logging()
    # 2-month smoke test
    logger.info("GDELT smoke test: NVDA 2014-01..02")
    news = fetch_news_for_ticker("NVDA","2014-01-01","2014-02-28")
    prices = fetch_prices("NVDA","2013-12-01","2014-03-31")
    labeled = label_with_returns(news, prices, HORIZONS)
    logger.info("news=%d, labeled=%d", len(news), len(labeled))
    if not labeled.empty:
        labeled.head(20).to_csv(os.path.join(OUT_DIR,"NVDA_2014_01_02_gdelt_sample.csv"), index=False)
        logger.info("Wrote data/NVDA_2014_01_02_gdelt_sample.csv")
    if not RUN_FULL:
        logger.info("Full crawl OFF (set RUN_FULL=True to run)"); raise SystemExit(0)

    all_frames=[]
    for tkr in COMPANY_ALIASES.keys():
        logger.info("=== %s GDELT %s..%s ===", tkr, NEWS_START, NEWS_END)
        n = fetch_news_for_ticker(tkr, NEWS_START, NEWS_END)
        p = fetch_prices(tkr, PRICE_START, PRICE_END)
        logger.info("news rows: %d, price rows: %d", len(n), len(p))
        if n.empty or p.empty:
            logger.warning("Skipping %s — insufficient data", tkr); continue
        lab = label_with_returns(n,p,HORIZONS); logger.info("labeled rows: %d", len(lab))
        lab.to_csv(os.path.join(OUT_DIR,f"{tkr}_gdelt_labeled.csv"), index=False)
        all_frames.append(lab)
    if all_frames:
        pd.concat(all_frames, ignore_index=True).sort_values(["ticker","date"]).to_csv(os.path.join(OUT_DIR,"all_gdelt_labeled.csv"), index=False)
        logger.info("GDELT combined written to data/all_gdelt_labeled.csv")
    else:
        logger.warning("GDELT produced no labeled data — broaden terms or try later")

