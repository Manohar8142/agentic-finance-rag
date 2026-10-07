"""Second-stage re-ranking of vector-search candidates."""
from __future__ import annotations

import math
from collections import Counter

from . import config
from .embeddings import tokenize
from .ingest import Chunk

STOP = set("the a an of and or to in for on is was were what how did does do with by at from as it its this that be are".split())


class BM25Reranker:
    """Blend BM25 over the candidate set with the dense score (hybrid re-ranking)."""

    def __init__(self, k1: float = 1.4, b: float = 0.75, alpha: float = 0.6) -> None:
        self.k1, self.b, self.alpha = k1, b, alpha

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 4) -> list[tuple[Chunk, float]]:
        if not candidates:
            return []
        q = [t for t in tokenize(query) if t not in STOP]
        docs = [tokenize(f"{c.meta.get('title', '')} {c.section} {c.text}") for c, _ in candidates]
        avg = sum(map(len, docs)) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        bm25 = []
        for d in docs:
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / avg))
            bm25.append(s)
        top = max(bm25) or 1.0
        dense = [s for _, s in candidates]
        lo, hi = min(dense), max(dense)
        scored = [
            (c, self.alpha * (b / top) + (1 - self.alpha) * ((s - lo) / (hi - lo) if hi > lo else 1.0))
            for (c, s), b in zip(candidates, bm25)
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]


class CrossEncoderReranker:
    def __init__(self, model: str = config.RERANKER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model)

    def rerank(self, query: str, candidates: list[tuple[Chunk, float]], top_k: int = 4) -> list[tuple[Chunk, float]]:
        scores = self._model.predict([(query, c.text) for c, _ in candidates])
        return sorted(((c, float(s)) for (c, _), s in zip(candidates, scores)), key=lambda x: x[1], reverse=True)[:top_k]


def get_reranker(kind: str | None = None):
    return CrossEncoderReranker() if (kind or config.RERANKER) == "cross-encoder" else BM25Reranker()
