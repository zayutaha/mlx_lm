"""Web search and content scraping for MLX LM."""

import re
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
SEARCH_TIMEOUT = 10
SCRAPE_TIMEOUT = 15
MAX_SCRAPE_BYTES = 200_000


def search_web(query: str, num_results: int = 5) -> List[dict]:
    """Search DuckDuckGo and return list of {title, url, snippet}."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=num_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return results


def scrape_url(url: str) -> Optional[str]:
    """Fetch and extract readable text content from a URL."""
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT,
        )
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type:
            return None

        soup = BeautifulSoup(resp.content, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "button", "iframe", "noscript",
                         "svg", "canvas", "audio", "video"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)

        if len(text.encode("utf-8")) > MAX_SCRAPE_BYTES:
            text = text[:MAX_SCRAPE_BYTES]

        return text.strip() or None

    except Exception:
        return None
