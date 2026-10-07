"""Embedding backends. ``hashing`` is offline and dependency-free; ``sentence-transformers`` is the quality option."""
from __future__ import annotations

import hashlib
import re

import numpy as np

from . import config

TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def _stem(tok: str) -> str:
    """Very light plural stripping so 'risks' matches 'risk' and 'margins' matches 'margin'."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss") and not tok[-2].isdigit():
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in TOKEN.findall(text.lower())]


class HashingEmbedder:
    """Signed feature hashing of word unigrams + bigrams, L2-normalised.

    Not semantic like a neural model, but deterministic, fast and good enough for
    keyword-heavy financial questions. Swap in sentence-transformers for real use.
    """

    name = "hashing-1024-v2"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for feat in self._features(t):
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = config.EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype="float32")


def get_embedder(kind: str | None = None):
    kind = kind or config.EMBEDDINGS
    return SentenceTransformerEmbedder() if kind == "sentence-transformers" else HashingEmbedder()
