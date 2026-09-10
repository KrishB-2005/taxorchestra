"""Retrieval over the IRS rule corpus.

BM25, hand-rolled in ~50 lines. The corpus is a few dozen short rules, so a
sparse lexical ranker beats dense embeddings here on every axis that matters:
it needs no model, no 500 MB of torch, no index build step, and it is
deterministic — the same query returns the same rules in the same order, which
is what makes the validation agent's output reproducible in tests.

`ElasticStore` swaps in the same interface for deployments that already run
Elasticsearch and want the corpus alongside everything else.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from taxorchestra.knowledge.irs_kb import RULES, Rule

_TOKEN = re.compile(r"[a-z0-9]+")

# Okapi BM25 defaults; k1 controls term-frequency saturation, b the strength of
# the length normalisation.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Retrieved:
    rule: Rule
    score: float


class KnowledgeStore(Protocol):
    def search(self, query: str, k: int = 3) -> list[Retrieved]: ...


class BM25Store:
    """In-process BM25 over the bundled rule corpus."""

    name = "bm25"

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self.rules = rules if rules is not None else RULES
        self._docs = [tokenize(f"{r.text} {r.citation}") for r in self.rules]
        self._lengths = [len(d) for d in self._docs]
        self._avg_len = (sum(self._lengths) / len(self._docs)) if self._docs else 0.0
        self._tf = [Counter(d) for d in self._docs]

        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        n = len(self._docs)
        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored so that a
        # term appearing in every document contributes ~0 rather than negative.
        self._idf = {
            term: max(math.log((n - count + 0.5) / (count + 0.5) + 1.0), 0.0)
            for term, count in df.items()
        }

    def search(self, query: str, k: int = 3) -> list[Retrieved]:
        terms = tokenize(query)
        if not terms or not self._docs:
            return []

        scored: list[Retrieved] = []
        for index, rule in enumerate(self.rules):
            length = self._lengths[index] or 1
            tf = self._tf[index]
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = freq + K1 * (1 - B + B * length / (self._avg_len or 1))
                score += idf * (freq * (K1 + 1)) / denom
            if score > 0:
                scored.append(Retrieved(rule=rule, score=score))

        scored.sort(key=lambda r: (-r.score, r.rule.rule_id))
        return scored[:k]


class ElasticStore:
    """Same interface, backed by an existing Elasticsearch cluster."""

    name = "elasticsearch"

    def __init__(self, url: str, index: str = "irs-rules") -> None:
        try:
            from elasticsearch import Elasticsearch
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "ElasticStore needs the elasticsearch client: pip install elasticsearch"
            ) from exc
        self._client = Elasticsearch(url)
        self._index = index

    def search(self, query: str, k: int = 3) -> list[Retrieved]:
        response = self._client.search(
            index=self._index,
            query={"match": {"text": query}},
            size=k,
        )
        return [
            Retrieved(
                rule=Rule(
                    rule_id=hit["_source"]["rule_id"],
                    citation=hit["_source"]["citation"],
                    text=hit["_source"]["text"],
                ),
                score=hit["_score"],
            )
            for hit in response["hits"]["hits"]
        ]


def build_store(backend: str = "bm25", **kwargs) -> KnowledgeStore:
    if backend == "bm25":
        return BM25Store()
    if backend == "elasticsearch":
        return ElasticStore(**kwargs)
    raise ValueError(f"unknown knowledge backend {backend!r}")
