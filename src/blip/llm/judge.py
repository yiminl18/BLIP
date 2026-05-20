from __future__ import annotations
import re
import logging
from blip._types import Pair
from blip.llm.usage import Usage

logger = logging.getLogger(__name__)

_CLOSED_DOMAIN_YES_NO = {"yes", "no"}
_YEAR_RE = re.compile(r"^\d{4}$")
_SHORT_LIST_RE = re.compile(r"^[^,]{1,30}(,[^,]{1,30}){0,2}$")


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _is_closed_domain(ground_truth: str | None) -> bool:
    if ground_truth is None:
        return False
    gt = ground_truth.strip()
    if gt.lower() in _CLOSED_DOMAIN_YES_NO:
        return True
    if _YEAR_RE.match(gt):
        return True
    # short list: <= 3 comma-separated tokens
    parts = [p.strip() for p in gt.split(",")]
    if len(parts) <= 3 and all(len(p.split()) <= 3 for p in parts):
        return True
    return False


def _parse_bool(text: str) -> bool | None:
    t = text.strip().lower()
    first = t.split()[0] if t.split() else ""
    if first in ("true", "yes", "1"):
        return True
    if first in ("false", "no", "0"):
        return False
    return None


def equivalent(
    a: str,
    b: str,
    *,
    closed_domain: bool | None = None,
    llm_client=None,
    pair: "Pair | None" = None,
) -> tuple[bool, list[Usage]]:
    """Return (equiv, usages). Uses exact match for closed-domain, LLM judge otherwise."""
    usages: list[Usage] = []

    if closed_domain is None and pair is not None:
        closed_domain = _is_closed_domain(pair.ground_truth)

    if closed_domain:
        return _normalize(a) == _normalize(b), usages

    # LLM judge
    if llm_client is None:
        # fallback to exact match if no client
        return _normalize(a) == _normalize(b), usages

    for attempt in range(2):
        content, usage = llm_client.judge(a, b)
        usages.append(usage)
        result = _parse_bool(content)
        if result is not None:
            return result, usages
        logger.warning("Judge parse failure attempt %d: %r", attempt + 1, content)

    logger.warning("Judge defaulting to False after parse failures")
    return False, usages
