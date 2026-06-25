"""
Email Finder — finds contact emails for companies via DuckDuckGo + website heuristics.
"""

import re
import time
from duckduckgo_search import DDGS

_LAST_SEARCH_TIME: float = 0.0
_MIN_INTERVAL: float = 1.2  # seconds between DuckDuckGo requests


def find_email_for_company(company_name: str, website: str = "", sector: str = "") -> str:
    """
    Tries to find a contact email for a company.
    1. Infer info@domain from website
    2. DuckDuckGo search
    Returns the best email found, or empty string if none.
    """
    # Step 1: Infer from website domain
    heuristic_email = ""
    if website:
        domain = (
            website.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .strip()
            .rstrip("/")
            .split("/")[0]
        )
        if "." in domain and " " not in domain:
            heuristic_email = f"info@{domain}"

    # Step 2: DuckDuckGo search for contact email
    try:
        # ── Guardrail: rate limiting — min 1.2s between requests ──
        global _LAST_SEARCH_TIME
        elapsed = time.time() - _LAST_SEARCH_TIME
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        _LAST_SEARCH_TIME = time.time()

        query = f'"{company_name}" contact email'
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        for r in results:
            text = r.get("body", "") + " " + r.get("href", "")
            found = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
            for e in found:
                e_lower = e.lower()
                if not any(x in e_lower for x in ["example", "test", "noreply", "no-reply", "sentry", ".png", ".jpg"]):
                    return e
    except Exception:
        pass

    return heuristic_email
