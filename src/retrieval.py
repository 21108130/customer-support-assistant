
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .corpus import Record, load_corpus


AUTHORITY_BOOST = {"policy": 0.06, "faq": 0.03, "ticket": 0.0}


@dataclass
class ScoredRecord:
    record: Record
    score: float
    raw_score: float


class Retriever:
    def __init__(self, records: Optional[list[Record]] = None):
        self.records = records if records is not None else load_corpus()
        corpus_text = [self._doc_text(r) for r in self.records]

        self.vectorizer = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, ngram_range=(1, 2)
        )
        self.matrix = self.vectorizer.fit_transform(corpus_text)

    @staticmethod
    def _doc_text(r: Record) -> str:
     
        return f"{r.title}. {r.title}. {r.text}"

    def _score_all(self, query: str) -> list[ScoredRecord]:
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        scored = []
        for rec, raw in zip(self.records, sims):
            boosted = float(raw) + AUTHORITY_BOOST.get(rec.source_type, 0.0)
            scored.append(ScoredRecord(record=rec, score=boosted, raw_score=float(raw)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored

    def search(self, query: str, k: int = 4, ensure_policy_coverage: bool = True) -> list[ScoredRecord]:
        """Return the top-k entries for `query`.

        `ensure_policy_coverage`: if the top-k window is dominated by
        tickets/FAQs but a policy entry scores at least half as well as the
        current top hit, swap in that policy. This implements the README's
        "a current policy should not be silently displaced by a similar-
        looking ticket" rule -- policies are the authoritative source for
        "what are the rules", tickets are only precedent/context.
        """
        ranked = self._score_all(query)
        if not ranked:
            return []
        top = ranked[:k]
        if ensure_policy_coverage and top:
            has_policy = any(s.record.source_type == "policy" for s in top)
            if not has_policy:
                best_policy = next(
                    (s for s in ranked if s.record.source_type == "policy"), None
                )
                if best_policy and best_policy.score >= top[0].score * 0.5:
                    top[-1] = best_policy
                    top.sort(key=lambda s: s.score, reverse=True)
        return top

    def top_score(self, query: str) -> float:
        ranked = self._score_all(query)
        return ranked[0].score if ranked else 0.0
