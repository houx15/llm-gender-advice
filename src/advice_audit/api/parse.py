"""Light-weight parsing of free-text advice: ranked-list extraction and refusal flags.

This is a structural check only (did the model give a ranked list?). Mapping to the
taxonomy is done by the LLM coder (advice_audit.coding).
"""
from __future__ import annotations

import json
import re

REFUSAL_PATTERNS = [
    r"\bI can(?:not|'t|’t) (?:help|provide|recommend|assist)",
    r"\bI(?:'m| am) (?:unable|not able) to\b",
    r"\bI (?:won't|will not) (?:recommend|provide)",
    r"\bas an AI\b",
    r"我无法(?:提供|推荐|回答)",
    r"我不能(?:提供|推荐|回答)",
    r"作为(?:一个)?(?:AI|人工智能)",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

# "1. X", "1) X", "**1. X**", "### 1. X", "1、X", "第一，X", "一、X"
_ITEM_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*(?:(?P<num>[1-9]|10)\s*[.)、:：]|第(?P<zh1>[一二三四五六七八九十])[、，,:：.]?|(?P<zh2>[一二三四五六七八九十])[、.])\s*(?P<text>.+)$"
)
_ZH_NUM = {c: i + 1 for i, c in enumerate("一二三四五六七八九十")}


def _clean(t: str) -> str:
    t = re.sub(r"[*_`#]+", "", t).strip()
    # headline = text before the first explanation separator
    t = re.split(r"\s+[–—-]\s+|[:：]\s*|——|：", t, maxsplit=1)[0].strip()
    return t.strip(" .。;；,，")


def extract_ranked(text: str, max_items: int = 5) -> list[dict]:
    items: list[dict] = []
    expected = 1
    for line in (text or "").splitlines():
        m = _ITEM_RE.match(line)
        if not m:
            continue
        n = int(m.group("num")) if m.group("num") else _ZH_NUM[m.group("zh1") or m.group("zh2")]
        if n != expected:
            continue  # nested or unrelated numbering
        items.append({"rank": n, "headline": _clean(m.group("text"))})
        expected += 1
        if len(items) >= max_items:
            break
    return items


def is_refusal(text: str, n_items: int) -> bool:
    if not text or not text.strip():
        return False
    return n_items == 0 and bool(_REFUSAL_RE.search(text))


def parse_response(text: str) -> dict:
    items = extract_ranked(text)
    return {
        "ranked_items": json.dumps(items, ensure_ascii=False),
        "n_items": len(items),
        "parse_ok": len(items) >= 1,
        "refusal_flag": is_refusal(text, len(items)),
    }


def parse_json_object(text: str) -> dict | None:
    """Parse a JSON object from a coder answer (tolerates code fences / leading prose)."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except json.JSONDecodeError:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            v = json.loads(s[i : j + 1])
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None
