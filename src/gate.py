
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .retrieval import ScoredRecord


CONFIDENCE_THRESHOLD = 0.12

SENSITIVE_PATTERNS = [
    r"\bchargeback\b", r"\bdispute(d)?\b.*\b(charge|transaction|payment)\b",
    r"\bfraud(ulent)?\b", r"\bunauthori[sz]ed\b",
    r"\bnever (bought|purchased|made|authorized)\b",
    r"\bstole|stolen\b", r"\blawsuit|legal action|attorney|lawyer\b",
    r"\bdelete my account\b", r"\bclose (my )?account permanently\b",
    r"\bfull card number|cvv|security code|ssn|social security\b",
    r"\bshare(d)? my password\b", r"\bmy password is\b",
]
SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


_RECOGNIZE_NEGATION_RE = re.compile(r"\b(don'?t|do not|didn'?t|did not) recognize\b", re.IGNORECASE)
_PAYMENT_WORD_RE = re.compile(r"\b(charge|charged|transaction|payment|card)\b", re.IGNORECASE)


def _mentions_unrecognized_payment(text: str) -> bool:
    return bool(_RECOGNIZE_NEGATION_RE.search(text) and _PAYMENT_WORD_RE.search(text))


STALE_QUERY_HINTS = re.compile(
    r"\b(outdated|old article|old policy|contradicts?|doesn't match|does not match|"
    r"used to say|says .* but|inconsistent)\b",
    re.IGNORECASE,
)


class Route(str, Enum):
    ANSWER = "answer"
    ESCALATE_LOW_CONFIDENCE = "escalate_low_confidence"
    ESCALATE_SENSITIVE = "escalate_sensitive"
    ESCALATE_STALE_CONFLICT = "escalate_stale_conflict"


@dataclass
class GateDecision:
    route: Route
    reason: str
    top_score: float
    flagged_stale_sources: list[str]


def decide(query: str, results: list[ScoredRecord]) -> GateDecision:
    top_score = results[0].score if results else 0.0
    stale_sources = [
        s.record.id for s in results if s.record.has_deprecated_note
    ]

    if SENSITIVE_RE.search(query) or _mentions_unrecognized_payment(query):
        return GateDecision(
            route=Route.ESCALATE_SENSITIVE,
            reason=(
                "The message involves a chargeback/fraud claim, an unrecognized "
                "or disputed charge, account deletion, legal language, or "
                "payment/credential details. These require human verification "
                "and must never be handled by an automated answer."
            ),
            top_score=top_score,
            flagged_stale_sources=stale_sources,
        )

    if not results or top_score < CONFIDENCE_THRESHOLD:
        return GateDecision(
            route=Route.ESCALATE_LOW_CONFIDENCE,
            reason=(
                f"Top retrieval score ({top_score:.3f}) is below the "
                f"confidence threshold ({CONFIDENCE_THRESHOLD}); nothing in "
                "the knowledge base is a confident match for this question."
            ),
            top_score=top_score,
            flagged_stale_sources=stale_sources,
        )

    if stale_sources and STALE_QUERY_HINTS.search(query):
        return GateDecision(
            route=Route.ESCALATE_STALE_CONFLICT,
            reason=(
                "The user is pointing at a discrepancy between current and "
                f"superseded wording, and {', '.join(stale_sources)} contains "
                "a known deprecated/contradicted claim. Routing to a human "
                "rather than letting the model adjudicate which version applies."
            ),
            top_score=top_score,
            flagged_stale_sources=stale_sources,
        )

    return GateDecision(
        route=Route.ANSWER,
        reason="Confident retrieval, no sensitive-intent or stale-conflict trigger.",
        top_score=top_score,
        flagged_stale_sources=stale_sources,
    )
