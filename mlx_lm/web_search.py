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


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract clean text from the main content area of a page."""
    candidates = []

    for sel in ("article", 'main', '[role="main"]', ".mw-parser-output",
                ".post-content", ".entry-content", ".article-body",
                "#article", "#content", ".content", "#mw-content-text"):
        els = soup.select(sel)
        for el in els:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 200:
                candidates.append(text)

    if candidates:
        return max(candidates, key=len)

    body = soup.find("body") or soup
    return body.get_text(separator="\n", strip=True)


def _clean_text(text: str) -> str:
    """Normalize whitespace and strip boilerplate lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    lines = text.splitlines()
    cleaned = []
    skip_to_end = False

    # Wikipedia-like sections that signal end of useful content
    end_section_triggers = re.compile(
        r"^(references|further reading|external links|"
        r"political offices|party political offices|"
        r"bibliography|notes|footnotes|"
        r"see also|sources|works cited|"
        r"categories|hidden categories)",
        re.I,
    )

    for line in lines:
        stripped = line.strip()

        # Once we hit References/Categories/etc, skip everything after
        if end_section_triggers.match(stripped):
            skip_to_end = True
            continue

        if skip_to_end:
            continue

        # Skip navigation/cookie boilerplate
        if re.match(
            r"^(skip to|jump to|navigation|main menu|search|"
            r"tools|what links here|related changes|special pages|"
            r"permanent link|page information|cite this page|"
            r"wikidata item|download as pdf|print/export|"
            r"cookie statement|privacy policy|about|"
            r"disclaimers|contact us|terms of use|"
            r"the free encyclopedia|this page was last edited|"
            r"retrieved from|text is available under|"
            r"by using this site|"
            r"short description is different from wikidata|"
            r"use dmy dates|articles containing|"
            r"from wikipedia|from wikiped|"
            r"stay connected|featured e.?books|sponsor content|"
            r"insights & reports|share this|next story|"
            r"editor.?s note|click to|read more|related stories|"
            r"trending now|most popular|you may also like|"
            r"advertisement|promoted|sponsored|"
            r"all rights reserved|privacy settings|"
            r"terms of service|terms and conditions)",
            stripped,
            re.I,
        ):
            continue

        # Skip lines that are just punctuation/symbols
        if re.match(r"^[\d\-–—|/\\*•·○●■□▼▲◆◇★☆♪♫]+$", stripped):
            continue

        if len(stripped) < 3:
            continue

        if stripped.startswith("Jump to") or stripped.startswith("Contents"):
            continue

        cleaned.append(stripped)

    return "\n".join(cleaned)


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
                         "svg", "canvas", "audio", "video", "menu"]):
            tag.decompose()

        text = _extract_main_content(soup)
        text = _clean_text(text)

        return text.strip() or None

    except Exception:
        return None


def is_relevant(title: str, snippet: str, original: str) -> bool:
    """Check if a search result is relevant to the original query by keyword matching."""
    keywords = set(original.lower().split())
    stopwords = {"what", "who", "where", "when", "why", "how", "the",
                 "and", "for", "are", "was", "were", "has", "had",
                 "did", "does", "do", "is", "of", "to", "in", "on",
                 "at", "by", "with", "from", "that", "this", "its"}
    keywords = {k for k in keywords if len(k) >= 3 and k not in stopwords}
    if not keywords:
        return True
    combined = (title + " " + snippet).lower()
    matches = sum(1 for k in keywords if k in combined)
    return matches >= 1 or (len(keywords) > 0 and matches / len(keywords) >= 0.3)
