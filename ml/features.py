"""
ml/features.py
URL feature extraction for the XGBoost model.

42 features across four buckets:
  1. Lexical    - length, character class counts, entropy, n-grams
  2. Host       - tld, subdomain depth, IP literal, port, scheme
  3. Path/query - depth, query param count, file extensions
  4. Heuristic  - brand keywords, suspicious tlds, shorteners, homoglyphs
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

# ----------------------------------------------------------------- constants

SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "loan",
    "work", "country", "stream", "download", "racing", "win",
    "review", "party", "trade", "bid", "date", "faith", "men",
    "kim", "icu", "buzz", "rest", "fit", "cyou",
})

SHORTENERS = frozenset({
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "t.co", "lnkd.in", "tiny.cc", "shorte.st", "rebrand.ly",
    "cutt.ly", "shorturl.at", "rb.gy", "tr.im",
})

BRAND_KEYWORDS = (
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook",
    "amazon", "google", "gmail", "facebook", "instagram", "whatsapp",
    "netflix", "spotify", "bank", "wellsfargo", "chase", "citibank",
    "barclays", "hsbc", "santander", "login", "secure", "account",
    "update", "verify", "signin", "webscr", "alert", "confirm",
    "support", "billing", "invoice", "wallet", "crypto", "coinbase",
)

LOGIN_HINTS = ("login", "signin", "verify", "account", "auth", "sso",
               "secure", "session", "wallet")

# Common homoglyph swaps (latin to look-alikes)
HOMOGLYPHS = {
    "0": "o", "1": "l", "rn": "m", "vv": "w", "5": "s",
}


# ----------------------------------------------------------------- helpers

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def is_ip_literal(host: str) -> int:
    if not host:
        return 0
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        return 1
    if ":" in host and re.search(r"[0-9a-f]+:[0-9a-f]+", host.lower()):
        return 1
    return 0


def has_homoglyph(s: str) -> int:
    s = s.lower()
    return int(any(h in s for h in HOMOGLYPHS))


def char_class_ratios(s: str) -> tuple[float, float, float]:
    if not s:
        return 0.0, 0.0, 0.0
    n = len(s)
    digits = sum(c.isdigit() for c in s)
    letters = sum(c.isalpha() for c in s)
    specials = n - digits - letters
    return digits / n, letters / n, specials / n


# ----------------------------------------------------------------- core API

def extract_features(url: str) -> dict:
    """Return a flat dict of 42 numeric features for one URL."""
    if not isinstance(url, str) or not url:
        url = ""
    if "://" not in url:
        url = "http://" + url

    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
    except Exception:
        parsed = urlparse("")
        ext = tldextract.extract("")

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.lower()

    digit_r, letter_r, special_r = char_class_ratios(url)

    return {
        # ---- Lexical (15)
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_at": url.count("@"),
        "num_question": url.count("?"),
        "num_equals": url.count("="),
        "num_underscore": url.count("_"),
        "num_ampersand": url.count("&"),
        "num_percent": url.count("%"),
        "num_digits": sum(c.isdigit() for c in url),
        "url_entropy": shannon_entropy(url),
        "host_entropy": shannon_entropy(host),

        # ---- Char class ratios (3)
        "digit_ratio": digit_r,
        "letter_ratio": letter_r,
        "special_ratio": special_r,

        # ---- Host (8)
        "subdomain_count": (
            len(ext.subdomain.split(".")) if ext.subdomain else 0),
        "tld_length": len(ext.suffix),
        "domain_length": len(ext.domain),
        "is_ip": is_ip_literal(host),
        "has_port": int(parsed.port is not None),
        "uses_https": int(parsed.scheme == "https"),
        "is_suspicious_tld": int(ext.suffix.split(".")[-1] in SUSPICIOUS_TLDS),
        "is_shortener": int(host in SHORTENERS),

        # ---- Path/query (8)
        "path_depth": path.count("/"),
        "double_slash_path": int("//" in path),
        "query_param_count": len(query.split("&")) if query else 0,
        "has_php": int(".php" in path.lower()),
        "has_html": int(".html" in path.lower() or ".htm" in path.lower()),
        "has_exe": int(".exe" in path.lower()),
        "has_zip": int(".zip" in path.lower()),
        "path_alpha_ratio": (
            sum(c.isalpha() for c in path) / max(len(path), 1)),

        # ---- Heuristic (8)
        "brand_in_path": int(any(b in path.lower() for b in BRAND_KEYWORDS)),
        "brand_in_subdomain": int(
            any(b in ext.subdomain.lower() for b in BRAND_KEYWORDS)),
        "brand_in_query": int(
            any(b in query.lower() for b in BRAND_KEYWORDS)),
        "brand_outside_domain": int(
            any(b in full for b in BRAND_KEYWORDS)
            and not any(b in ext.domain.lower() for b in BRAND_KEYWORDS)),
        "has_login_hint": int(any(k in full for k in LOGIN_HINTS)),
        "has_homoglyph": has_homoglyph(host),
        "tld_in_path": int(
            any(f".{t} " in path or f".{t}/" in path
                for t in ["com", "net", "org"])),
        "punycode": int("xn--" in host),
    }


def feature_names() -> list[str]:
    return list(extract_features("http://example.com/").keys())


def features_to_vector(url: str) -> list[float]:
    return [float(v) for v in extract_features(url).values()]
