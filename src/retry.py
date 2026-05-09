import logging
import time

import feedparser

logger = logging.getLogger(__name__)

# Returned when all retries are exhausted; .entries is [] so callers can
# safely iterate without a null-check.
_EMPTY_FEED = feedparser.FeedParserDict({"entries": [], "bozo": False})


def fetch_feed(url: str, *, max_retries: int, base_backoff: float) -> feedparser.FeedParserDict:
    """Fetch an RSS feed with exponential-backoff retries.

    Treats both raised exceptions and feedparser bozo errors (bozo=True with
    no entries) as retryable failures.  On final failure logs a warning and
    returns an empty FeedParserDict so callers can safely iterate feed.entries.

    Args:
        url: RSS feed URL.
        max_retries: Number of retry attempts after the initial try (total
            attempts = max_retries + 1).
        base_backoff: Base delay in seconds; actual delays are base_backoff,
            base_backoff*2, base_backoff*4, ...
    """
    for attempt in range(max_retries + 1):
        try:
            feed = feedparser.parse(url)
            if feed.bozo and not feed.entries:
                exc: Exception = getattr(feed, "bozo_exception", None) or Exception(
                    "feedparser bozo error with no entries"
                )
                raise exc
            return feed
        except Exception as e:
            if attempt == max_retries:
                logger.warning("RSS feed %s failed after %d attempt(s): %s", url, max_retries + 1, e)
                return _EMPTY_FEED
            sleep_secs = base_backoff * (2 ** attempt)
            logger.debug(
                "RSS feed %s attempt %d failed, retrying in %.1fs: %s",
                url, attempt + 1, sleep_secs, e,
            )
            time.sleep(sleep_secs)

    return _EMPTY_FEED  # unreachable; satisfies type checkers
