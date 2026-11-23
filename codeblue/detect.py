"""Find sensitive spans locally, before anything is sent to the cloud.

- RegexDetector: patterns for money, percentages, emails, phone/account numbers, dates, codes,
  large numbers, multi-word proper names, plus the organisation's own glossary of names.
- LLMDetector: asks the LOCAL model to list sensitive spans (PAAC-style). Its calls go to the local log.
- OracleDetector: uses the secret labels. Upper bound only; never use it for headline results.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

# Self-contained on purpose (stdlib only): this file is also copied into EC2-Bench's cloud proxy for P3.
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text or "").strip()


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    category: str


_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
PATTERNS: list[tuple[str, str]] = [
    ("EMAIL", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    ("MONEY", r"(?:S\$|US\$|RM|\$|SGD ?|USD ?|MYR ?)\d[\d,]*(?:\.\d+)?(?: ?(?:million|billion|bn|m|k)\b)?"),
    ("PCT", r"\b\d+(?:\.\d+)? ?%"),
    ("DATE", rf"\b\d{{1,2}} {_MONTH} \d{{4}}\b|\b{_MONTH} \d{{1,2}},? \d{{4}}\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b"),
    ("PHONE", r"\+\d{2,3}[\d -]{7,}\d"),
    ("ACCOUNT", r"\b\d{3,}-\d{4,}(?:-\d+)*\b"),
    ("CODE", r"\b(?:[A-Z]{3,} )*(?=[A-Z0-9-]*\d)(?=[A-Z0-9-]*[A-Z])[A-Z0-9]+(?:-[A-Z0-9]+)+\b"),
    ("NUM", r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{5,}(?:\.\d+)?\b"),
]

# Capitalised words that start many sentences or headings and are rarely sensitive on their own.
_COMMON = set(
    """The A An This That These Those We Our You Your It Its In On At For From To Of And Or But If When
    Total Board Q1 Q2 Q3 Q4 Managers Staff Questions Decision Target Deal Project Margins Customer All Do
    Not Advisers Exclusivity Competitor Notes Index Warehouse Restructuring Plan Agenda CFO HQ Air Ocean
    Trucking Reefer Full Cold Singapore""".split()
)
_PROPER = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")


class RegexDetector:
    name = "regex"

    def __init__(self, glossary: list[str] | None = None, min_proper_words: int = 2):
        self.glossary = sorted(glossary or [], key=len, reverse=True)
        self.min_proper_words = min_proper_words

    def find(self, text: str) -> list[Span]:
        spans: list[Span] = []
        for term in self.glossary:
            for m in re.finditer(rf"\b{re.escape(term)}\b", text, re.I):
                spans.append(Span(m.start(), m.end(), m.group(0), "NAME"))
        for cat, pat in PATTERNS:
            for m in re.finditer(pat, text, re.I if cat in ("MONEY", "DATE") else 0):
                spans.append(Span(m.start(), m.end(), m.group(0), cat))
        for m in _PROPER.finditer(text):
            words = m.group(0).split()
            if len(words) >= self.min_proper_words and not all(w in _COMMON for w in words):
                spans.append(Span(m.start(), m.end(), m.group(0), "NAME"))
        return resolve_overlaps(spans)


class LLMDetector:
    """Ask the local model for sensitive spans. Falls back to regex when the reply is unusable."""

    name = "llm"
    PROMPT = (
        "List every confidential span in the text below: names of companies and people, amounts, percentages, "
        "dates, codes, account numbers, email addresses, and any business plan details. Reply with a JSON list "
        'like [{"text": "...", "type": "NAME|MONEY|PCT|DATE|CODE|ACCOUNT|EMAIL|OTHER"}] and nothing else.'
    )

    def __init__(self, local_model=None, glossary: list[str] | None = None):
        self.model = local_model
        self.fallback = RegexDetector(glossary)

    def messages(self, text: str) -> list[dict]:
        return [{"role": "system", "content": self.PROMPT}, {"role": "user", "content": text}]

    def find(self, text: str) -> list[Span]:
        return self.parse(self.model.complete(self.messages(text), "detect"), text)

    def parse(self, reply: str, text: str) -> list[Span]:
        """Turn the model's reply into spans of `text`; regex spans are always added; regex only if unusable."""
        reply = strip_thinking(reply)
        m = re.search(r"\[.*\]", reply, re.S)
        try:
            items = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            items = None
        if not isinstance(items, list):
            return self.fallback.find(text)
        spans = []
        for it in items:
            if not isinstance(it, dict) or not str(it.get("text", "")).strip():
                continue
            needle = str(it["text"]).strip()
            cat = re.sub(r"[^A-Z]", "", str(it.get("type", "OTHER")).upper()) or "OTHER"
            for mm in re.finditer(re.escape(needle), text):
                spans.append(Span(mm.start(), mm.end(), mm.group(0), cat))
        return resolve_overlaps(spans + self.fallback.find(text))


class OracleDetector:
    """Uses the labelled secrets. Only for an upper-bound ablation."""

    name = "oracle"

    def __init__(self, secrets, glossary: list[str] | None = None):
        terms: set[str] = set()
        for s in secrets.values():
            terms.update(v for v in [s.value, *s.variants] if v)
            for group in s.keywords:
                terms.update(t for t in group if len(t) >= 4)
        self.inner = RegexDetector(sorted(terms) + list(glossary or []))

    def find(self, text: str) -> list[Span]:
        return self.inner.find(text)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Keep the longest span where spans overlap (ties: earliest)."""
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if all(s.end <= o.start or s.start >= o.end for o in out):
            out.append(s)
    return sorted(out, key=lambda s: s.start)


def make_detector(kind: str, room, local_model=None):
    if kind == "regex":
        return RegexDetector(room.glossary)
    if kind == "llm":
        if local_model is None:
            raise ValueError("the llm detector needs the local model")
        return LLMDetector(local_model, room.glossary)
    if kind == "oracle":
        return OracleDetector(room.secrets, room.glossary)
    raise ValueError(f"unknown detector {kind!r}")
