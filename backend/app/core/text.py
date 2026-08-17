"""Shared text utilities: token estimation, normalisation and lexical overlap.

``estimate_tokens`` is deliberately a cheap heuristic. Wherever a provider
reports real usage we record the reported number instead (see
``app.llm.gateway``); the estimate is only used for chunk sizing and for
budgeting before a call is made.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[A-Za-z0-9_]+(?:['\-][A-Za-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

STOPWORDS = {
    "a", "about", "above", "after", "again", "all", "also", "am", "an", "and", "any", "are", "as",
    "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own",
    "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why",
    "will", "with", "would", "you", "your",
}


def estimate_tokens(text: str) -> int:
    """Approximate token count (~4 characters per token, word-count floor)."""
    if not text:
        return 0
    return max(1, int(len(text) / 4), len(text.split()) // 2)


def normalize_whitespace(text: str) -> str:
    text = text.replace(" ", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = [t.lower() for t in _WORD.findall(text or "")]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = _SENTENCE.split(normalize_whitespace(text))
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def token_overlap(query: str, text: str) -> float:
    """Fraction of the query's content tokens that appear in ``text``."""
    q = set(tokenize(query))
    if not q:
        return 0.0
    t = set(tokenize(text))
    return len(q & t) / len(q)


def cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else 0.0


def lexical_similarity(a: str, b: str) -> float:
    """TF cosine similarity over content words -- a fast, embedding-free proxy."""
    return cosine_counter(Counter(tokenize(a)), Counter(tokenize(b)))


def truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def extract_quoted_phrases(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r'"([^"]{2,80})"', text or "") if m.strip()]


ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z0-9]+)*\b")
IDENTIFIER = re.compile(r"\b(?:[A-Za-z]+[-_]?\d+[A-Za-z0-9\-_.]*|[A-Za-z]+\d[A-Za-z0-9]*)\b")
CAMEL_CASE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")


def technical_terms(text: str) -> list[str]:
    """Acronyms, model identifiers and CamelCase names -- the tokens that make
    a query keyword-sensitive and therefore favour BM25 over dense retrieval."""
    found: list[str] = []
    for pattern in (ACRONYM, IDENTIFIER, CAMEL_CASE):
        found.extend(pattern.findall(text or ""))
    seen: set[str] = set()
    out: list[str] = []
    for term in found:
        if term.lower() in STOPWORDS or len(term) < 2:
            continue
        if term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return out


__all__ = [
    "estimate_tokens",
    "normalize_whitespace",
    "tokenize",
    "split_sentences",
    "jaccard",
    "token_overlap",
    "lexical_similarity",
    "truncate_words",
    "extract_quoted_phrases",
    "technical_terms",
    "STOPWORDS",
]
