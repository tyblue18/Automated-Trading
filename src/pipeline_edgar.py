# src/pipeline_edgar.py
# EDGAR ingest: submissions (+yearly) + search-index (chunked, 2-pass) + HTML scraper
# If windowed rows are 0 after submissions/search-index, auto-scrape HTML and union.
# Robust schema normalization, safe doc_url building, 3d/5d return labels.

import os, re, time, datetime as dt, requests
import pandas as pd
import yfinance as yf
from bs4 import BeautifulSoup
from typing import List, Optional, Tuple
import logging

# ========= Config =========
OUT_DIR = "data"; os.makedirs(OUT_DIR, exist_ok=True)

TICKER_CIK = {
    "NVDA": "0001045810",
    "AMD":  "0000002488",
    "TSM":  "0001046179",
}

# Base forms (amendments like 8-K/A are kept via base-form normalization)
BASE_FORMS = {"8-K", "10-Q", "10-K", "6-K"}

# Filing window you care about
START = "2008-01-01"
END   = "2018-12-31"

# Price window (pad around START/END for lookahead)
PRICE_START = "2007-12-01"
PRICE_END   = "2019-01-31"

# Returns horizons (calendar days → next trading day)
HORIZONS = [3, 5]

# Cap text fetches per ticker (raise to get more)
MAX_DOCS = 500

# SEC requires a UA — use a real email
UA = {"User-Agent": "stock-news-llm/0.1 (contact: you@example.com)"}

# Search-index chunking (query smaller windows to reach older years reliably)
CHUNK_YEARS = 3      # try 3-year chunks between START and END
SEARCH_PAGES = 10    # pages per chunk per query (each page returns up to 'size' rows)
SEARCH_SIZE  = 200   # rows per page

# ========= HTTP helpers =========
logger = logging.getLogger(__name__)

def _get_json(url: str, method: str = "GET", payload: Optional[dict] = None,
              max_retries: int = 4, sleep_base: float = 0.7, timeout: int = 30):
    """
    Fetch JSON data from URL with exponential backoff retry logic.
    
    Args:
        url: URL to fetch
        method: HTTP method ("GET" or "POST")
        payload: JSON payload for POST requests
        max_retries: Maximum number of retry attempts
        sleep_base: Base sleep time for exponential backoff
        timeout: Request timeout in seconds
    
    Returns:
        JSON data as dict, or None if all retries failed
    """
    for i in range(max_retries):
        try:
            if method == "POST":
                r = requests.post(url, headers=UA, json=payload, timeout=timeout)
            else:
                r = requests.get(url, headers=UA, timeout=timeout)
            
            if r.ok:
                return r.json()
            else:
                logger.warning(f"HTTP {r.status_code} for {url} (attempt {i+1}/{max_retries})")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {url} (attempt {i+1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {url}: {e} (attempt {i+1}/{max_retries})")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e} (attempt {i+1}/{max_retries})")
        
        if i < max_retries - 1:  # Don't sleep after last attempt
            sleep_time = sleep_base * (2 ** i)
            logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None

def _get_html(url: str, max_retries: int = 4, sleep_base: float = 0.7, timeout: int = 30) -> Optional[str]:
    """
    Fetch HTML content from URL with exponential backoff retry logic.
    
    Args:
        url: URL to fetch
        max_retries: Maximum number of retry attempts
        sleep_base: Base sleep time for exponential backoff
        timeout: Request timeout in seconds
    
    Returns:
        HTML content as string, or None if all retries failed
    """
    for i in range(max_retries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.ok:
                return r.text
            else:
                logger.warning(f"HTTP {r.status_code} for {url} (attempt {i+1}/{max_retries})")
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {url} (attempt {i+1}/{max_retries})")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {url}: {e} (attempt {i+1}/{max_retries})")
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e} (attempt {i+1}/{max_retries})")
        
        if i < max_retries - 1:  # Don't sleep after last attempt
            sleep_time = sleep_base * (2 ** i)
            logger.debug(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
    
    logger.error(f"Failed to fetch {url} after {max_retries} attempts")
    return None

# ========= Schema normalization =========
def _coalesce_datetime(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    for c in candidates:
        if c in df.columns:
            out = out.fillna(pd.to_datetime(df[c], errors="coerce"))
    return out

def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()

    # filingDate may be filed/dateFiled/accepted/reportDate
    if "filingDate" not in df.columns or df["filingDate"].isna().all():
        df["filingDate"] = _coalesce_datetime(df, ["filingDate","filed","dateFiled","accepted","reportDate"])
    else:
        df["filingDate"] = pd.to_datetime(df["filingDate"], errors="coerce")

    # form may appear under other names
    if "form" not in df.columns:
        for alt in ["formType","forms","documentType"]:
            if alt in df.columns:
                df["form"] = df[alt].astype(str); break
    if "form" not in df.columns: df["form"] = ""
    df["form"] = df["form"].fillna("").astype(str)

    for c in ["primaryDocument","primaryDocDescription","accessionNumber","doc_url"]:
        if c not in df.columns: df[c] = ""
    return df

def _base_form(s: str) -> str:
    s = (s or "").strip().upper()
    return s[:-2] if s.endswith("/A") else s

# ========= Submissions path =========
def _load_company_submissions(cik: str) -> Optional[dict]:
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    return _get_json(url)

def _recent_filings_df(sub: dict) -> pd.DataFrame:
    rec = sub.get("filings", {}).get("recent", {})
    df = pd.DataFrame(rec)
    if df.empty: return df
    df["filingDate"] = pd.to_datetime(df.get("filingDate"), errors="coerce")
    if "reportDate" in df.columns:
        df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    else:
        df["reportDate"] = pd.NaT
    for c in ["primaryDocument","primaryDocDescription","accessionNumber","form"]:
        if c not in df.columns: df[c] = ""
    return df

def _load_year_file(name: str) -> pd.DataFrame:
    url = f"https://data.sec.gov/submissions/{name}"
    j = _get_json(url)
    if not j: return pd.DataFrame()
    rows = j.get("filings", [])
    df = pd.DataFrame(rows)
    if df.empty: return df
    df["filingDate"] = pd.to_datetime(df.get("filingDate"), errors="coerce") if "filingDate" in df.columns else pd.NaT
    df["reportDate"] = pd.to_datetime(df.get("reportDate"), errors="coerce") if "reportDate" in df.columns else pd.NaT
    for c in ["primaryDocument","primaryDocDescription","accessionNumber","form"]:
        if c not in df.columns: df[c] = ""
    return df

# ========= Search-index (chunked) =========
def _year_chunks(start: str, end: str, span_years: int) -> List[Tuple[str,str]]:
    s = pd.Timestamp(start); e = pd.Timestamp(end)
    chunks = []
    cur = s
    while cur <= e:
        nxt = (cur + pd.DateOffset(years=span_years)) - pd.Timedelta(days=1)
        if nxt > e: nxt = e
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt + pd.Timedelta(days=1)
    return chunks

def _search_index_once(keys: List[str], ciks: List[str], start: str, end: str,
                       use_forms: bool, pages: int, size: int) -> pd.DataFrame:
    url = "https://efts.sec.gov/LATEST/search-index"
    forms = sorted(list(BASE_FORMS)) + [f"{f}/A" for f in BASE_FORMS] if use_forms else None
    frames = []

    # textual keys
    for key in keys:
        for page in range(pages):
            payload = {"keys": key, "category":"custom", "startdt":start, "enddt":end,
                       "from": page*size, "size": size}
            if use_forms: payload["forms"] = forms
            j = _get_json(url, method="POST", payload=payload)
            if not j: break
            hits = j.get("hits", {}).get("hits", [])
            if not hits: break
            frames.append(pd.json_normalize([h.get("_source", {}) for h in hits]))

    # explicit CIKs
    for cik in ciks:
        for page in range(pages):
            payload = {"ciks":[str(int(cik))], "category":"custom", "startdt":start, "enddt":end,
                       "from": page*size, "size": size}
            if use_forms: payload["forms"] = forms
            j = _get_json(url, method="POST", payload=payload)
            if not j: break
            hits = j.get("hits", {}).get("hits", [])
            if not hits: break
            frames.append(pd.json_normalize([h.get("_source", {}) for h in hits]))

    if not frames: return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["adsh"], keep="first")

    out = pd.DataFrame()
    out["filingDate"] = pd.to_datetime(raw.get("filed", ""), errors="coerce")
    out["form"] = raw.get("form", "").astype(str)
    out["accessionNumber"] = raw.get("adsh", "").astype(str)
    out["primaryDocument"] = raw.get("primary_document", "").astype(str)

    if "link" in raw.columns:
        out["doc_url"] = raw["link"].astype(str)
    else:
        cik_series = raw.get("cik")
        nodash = out["accessionNumber"].fillna("").str.replace("-", "", regex=False)
        prim = out["primaryDocument"].fillna("")
        cik_val = (cik_series.astype(str) if cik_series is not None else "")
        out["doc_url"] = "https://www.sec.gov/Archives/edgar/data/" + cik_val + "/" + nodash + "/" + prim
        out.loc[out["doc_url"].str.endswith("/"), "doc_url"] += "index.html"

    out["primaryDocDescription"] = (raw.get("display_names.0") or "")
    if "primaryDocDescription" in out.columns:
        out["primaryDocDescription"] = out["primaryDocDescription"].astype(str)
    else:
        out["primaryDocDescription"] = ""
    keep = ["filingDate","form","doc_url","accessionNumber","primaryDocument","primaryDocDescription"]
    for c in keep:
        if c not in out.columns: out[c] = ""
    return out.dropna(subset=["filingDate"]).reset_index(drop=True)

def _search_index_chunked(keys: List[str], ciks: List[str], start: str, end: str) -> pd.DataFrame:
    chunks = _year_chunks(start, end, CHUNK_YEARS)
    frames = []
    for cs, ce in chunks:
        df = _search_index_once(keys, ciks, cs, ce, True, SEARCH_PAGES, SEARCH_SIZE)
        if df.empty:
            df = _search_index_once(keys, ciks, cs, ce, False, SEARCH_PAGES, SEARCH_SIZE)
        if not df.empty:
            frames.append(df)
        time.sleep(0.2)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# ========= HTML scraper =========
def _scrape_company_filings_html(cik: str, start: str, end: str, base_forms=BASE_FORMS,
                                 max_pages: int = 12, count_per_page: int = 100) -> pd.DataFrame:
    def parse_table(html: str) -> List[dict]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table", class_="tableFile2")
        out = []
        if not table: return out
        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 5: continue
            form = tds[0].get_text(strip=True)
            if not form: continue
            base = _base_form(form)
            filing_href = tds[1].find("a")
            doc_href    = tds[1].find_all("a")[-1] if tds[1].find_all("a") else None
            date_str = tds[3].get_text(strip=True)
            fdate = pd.to_datetime(date_str, errors="coerce")
            doc_url = ""
            if doc_href and doc_href.get("href"):
                u = doc_href.get("href")
                doc_url = "https://www.sec.gov" + u if u.startswith("/") else u
            elif filing_href and filing_href.get("href"):
                u = filing_href.get("href")
                doc_url = ("https://www.sec.gov" + u + "index.html") if u.endswith("/") \
                          else ("https://www.sec.gov" + u.rsplit("/",1)[0] + "/index.html")
            out.append({
                "filingDate": fdate, "form": form, "form_base": base,
                "doc_url": doc_url, "accessionNumber": "", "primaryDocument": "", "primaryDocDescription": ""
            })
        return out

    frames = []
    for f in base_forms:
        for page in range(max_pages):
            start_idx = page * count_per_page
            url = ("https://www.sec.gov/cgi-bin/browse-edgar"
                   f"?action=getcompany&CIK={int(cik)}&type={f}&owner=exclude"
                   f"&count={count_per_page}&start={start_idx}")
            html = _get_html(url)
            if not html: break
            items = parse_table(html)
            if not items: break
            frames.append(pd.DataFrame(items))
            time.sleep(0.4)

    if not frames: return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = df.dropna(subset=["filingDate"])
    df = df[df["filingDate"].between(start, end)]
    df = df.drop_duplicates(subset=["doc_url"]).reset_index(drop=True)
    print(f"  [html-scraper] rows: {len(df)}")
    return df

# ========= Master loader =========
def load_filings_in_range(cik: str, company_key: str, start: str, end: str) -> pd.DataFrame:
    frames = []

    # 1) submissions (recent + yearly files)
    sub = _load_company_submissions(cik)
    if sub:
        recent = _recent_filings_df(sub)
        if not recent.empty: frames.append(recent)
        for f in sub.get("filings", {}).get("files", []):
            name = f.get("name")
            if name:
                yr = _load_year_file(name)
                if not yr.empty: frames.append(yr)

    # 2) chunked search-index for the exact years you want
    keys = [company_key, company_key.upper(), company_key.lower()]
    alias_map = {"NVDA":["NVIDIA","NVIDIA Corporation"],
                 "AMD":["AMD","Advanced Micro Devices"],
                 "TSM":["TSMC","Taiwan Semiconductor"]}
    keys += alias_map.get(company_key.upper(), [])
    si = _search_index_chunked(keys=keys, ciks=[str(int(cik))], start=start, end=end)
    if not si.empty: frames.append(si)

    df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if df.empty:
        # 3) if still empty before normalization, scrape HTML
        print("  (no rows from submissions/search-index — scraping classic EDGAR)")
        html_df = _scrape_company_filings_html(cik, start, end, base_forms=BASE_FORMS)
        df = html_df if not html_df.empty else pd.DataFrame()

    if df.empty: return df

    # Normalize BEFORE filtering
    df = _normalize_schema(df)

    # Coalesced window date, and apply target window
    df["window_date"] = _coalesce_datetime(df, ["filingDate","filed","dateFiled","reportDate","accepted"])
    df = df[~df["window_date"].isna()].copy()
    print(f"  rows before windowing: {len(df)}")
    print(f"  date span (min → max): {df['window_date'].min()} → {df['window_date'].max()}")

    df = df[df["window_date"].between(START, END)].copy()
    print(f"  rows after windowing: {len(df)}")

    # If window still empty, scrape HTML as a fallback and union
    if df.empty:
        print("  (window empty — pulling HTML fallback for target years)")
        html_df = _scrape_company_filings_html(cik, start, end, base_forms=BASE_FORMS)
        if not html_df.empty:
            html_df = _normalize_schema(html_df)
            html_df["window_date"] = _coalesce_datetime(html_df, ["filingDate","filed","dateFiled","reportDate","accepted"])
            html_df = html_df[~html_df["window_date"].isna()]
            html_df = html_df[html_df["window_date"].between(START, END)]
            df = html_df.copy()
            print(f"  rows after HTML fallback: {len(df)}")

    if df.empty:
        return df

    # Debug forms before base-filter
    pre_counts = df["form"].value_counts().sort_index()
    print("  pre-filter (raw) form counts:", ", ".join(f"{k}:{v}" for k,v in pre_counts.items()) if not pre_counts.empty else "(none)")

    # Base-form filter
    df["form_base"] = df["form"].astype(str).map(_base_form)
    df = df[df["form_base"].isin(BASE_FORMS)]
    if df.empty:
        print("  after base-form filtering: 0")
        return df

    # Ensure doc_url exists or build safely
    if "doc_url" not in df.columns: df["doc_url"] = ""
    doc_missing = df["doc_url"].astype(str).str.strip() == ""
    if doc_missing.any():
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}"
        acc  = df["accessionNumber"].astype(str) if "accessionNumber" in df.columns else pd.Series("", index=df.index)
        nod  = acc.fillna("").str.replace("-", "", regex=False)
        prim = df["primaryDocument"].astype(str) if "primaryDocument" in df.columns else pd.Series("", index=df.index)
        prim = prim.fillna("")
        built = base + "/" + nod + "/" + prim
        built = built.where(~built.str.endswith("/"), built + "index.html")
        df.loc[doc_missing, "doc_url"] = built.loc[doc_missing]

    # Final tidy & dedup
    for c in ["primaryDocDescription","primaryDocument","accessionNumber","doc_url"]:
        if c not in df.columns: df[c] = ""
    df = df.drop_duplicates(subset=["doc_url"]).reset_index(drop=True)

    counts_raw  = df["form"].value_counts().sort_index().to_dict()
    counts_base = df["form_base"].value_counts().sort_index().to_dict()
    print("  loaded filings:", len(df), "| by raw form:", counts_raw, " | by base form:", counts_base)
    return df

# ========= Text, prices, labeling =========
def fetch_text(url: str) -> str:
    """
    Fetch and extract text content from a URL.
    
    Args:
        url: URL to fetch text from
    
    Returns:
        Extracted text content, or empty string on error
    """
    try:
        r = requests.get(url, headers=UA, timeout=30)
        if not r.ok:
            logger.warning(f"HTTP {r.status_code} when fetching text from {url}")
            return ""
        
        soup = BeautifulSoup(r.text, "lxml")
        for tag in soup(["script","style","noscript"]): 
            tag.extract()
        
        text = soup.get_text("\n")
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error fetching text from {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error extracting text from {url}: {e}")
        return ""

def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch stock price data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
    
    Returns:
        DataFrame with price data, or empty DataFrame on error
    """
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            logger.warning(f"No price data found for {ticker} from {start} to {end}")
            return pd.DataFrame()
        
        df = df.reset_index().rename(columns={"Date":"date"})
        df["date"] = pd.to_datetime(df["date"])
        
        logger.info(f"Fetched {len(df)} price records for {ticker}")
        return df[["date","Open","High","Low","Close","Volume"]]
    except Exception as e:
        logger.error(f"Error fetching prices for {ticker}: {e}")
        return pd.DataFrame()

def first_on_or_after(prices: pd.DataFrame, dt0: pd.Timestamp):
    sub = prices[prices["date"] >= dt0]
    return None if sub.empty else sub.iloc[0]

def label_with_returns(rows: pd.DataFrame, prices: pd.DataFrame, horizons=(3,5)) -> pd.DataFrame:
    if rows.empty or prices.empty: return pd.DataFrame()
    out = []
    for _, r in rows.iterrows():
        d0 = r["filingDate"]
        row0 = first_on_or_after(prices, d0)
        if row0 is None: continue
        p0 = float(row0["Close"].item() if hasattr(row0["Close"], "item") else row0["Close"])
        rec = {
            "date": d0,
            "form": r.get("form_base") or r.get("form") or "",
            "url": r["doc_url"],
            "title": (r.get("primaryDocDescription") or r.get("primaryDocument") or "").strip(),
            "snippet": "",
            "price_t0": p0
        }
        for h in horizons:
            rowh = first_on_or_after(prices, d0 + pd.Timedelta(days=h))
            if rowh is None:
                rec[f"ret_{h}d"] = None; rec[f"label_{h}d"] = None
            else:
                pH = float(rowh["Close"].item() if hasattr(rowh["Close"], "item") else rowh["Close"])
                ret = pH / p0 - 1.0
                rec[f"ret_{h}d"] = ret
                rec[f"label_{h}d"] = "UP" if ret > 0 else "DOWN"
        out.append(rec)
    return pd.DataFrame(out).reset_index(drop=True)

# ========= Main =========
if __name__ == "__main__":
    all_frames = []

    for tkr, cik in TICKER_CIK.items():
        print(f"\n=== {tkr} EDGAR {START}..{END} ===")
        filings = load_filings_in_range(cik, company_key=tkr, start=START, end=END)
        if filings.empty:
            print("  (no filings after loaders)")
            continue

        # Fetch text (polite & capped)
        texts = []
        for i, (_, row) in enumerate(filings.iterrows()):
            if i >= MAX_DOCS: break
            texts.append(fetch_text(row["doc_url"]))
            time.sleep(0.5)
        filings = filings.iloc[:len(texts)].copy()
        filings["text"] = texts

        # Prices + labels
        prices = fetch_prices(tkr, PRICE_START, PRICE_END)
        if prices.empty:
            print("  (no prices)")
            continue

        labeled = label_with_returns(filings, prices, HORIZONS)
        labeled["snippet"] = [(filings.iloc[i]["text"] or "")[:2000] for i in range(len(labeled))]
        labeled.insert(0, "ticker", tkr)
        print("  labeled rows:", len(labeled))

        if not labeled.empty:
            out_path = os.path.join(OUT_DIR, f"{tkr}_edgar_labeled.csv")
            labeled.to_csv(out_path, index=False)
            all_frames.append(labeled)

    if all_frames:
        pd.concat(all_frames, ignore_index=True)\
          .sort_values(["ticker","date"])\
          .to_csv(os.path.join(OUT_DIR, "all_edgar_labeled.csv"), index=False)
        print("\n✅ EDGAR combined written to data/all_edgar_labeled.csv")
    else:
        print("\n⚠️ EDGAR produced no labeled data — raise MAX_DOCS, add tickers, or extend dates.")

