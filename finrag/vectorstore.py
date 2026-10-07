"""FAISS vector store (inner product over normalised vectors = cosine similarity), persisted to disk."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss

from .ingest import Chunk


class FaissStore:
    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.index: faiss.Index | None = None
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> None:
        vecs = self.embedder.embed([f"{c.meta.get('title', '')} | {c.section}. {c.text}" for c in chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend(chunks)

    def search(self, query: str, k: int = 12) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        q = self.embedder.embed([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        meta = {"embedder": self.embedder.name, "chunks": [asdict(c) for c in self.chunks]}
        (path / "chunks.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, embedder) -> "FaissStore":
        meta = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        if meta["embedder"] != embedder.name:
            raise ValueError(f"index built with {meta['embedder']}, current embedder is {embedder.name}; rebuild it")
        store = cls(embedder)
        store.index = faiss.read_index(str(path / "index.faiss"))
        store.chunks = [Chunk(**c) for c in meta["chunks"]]
        return store

    @staticmethod
    def exists(path: Path) -> bool:
        return (path / "index.faiss").exists() and (path / "chunks.json").exists()

