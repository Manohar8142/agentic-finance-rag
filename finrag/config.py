from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data" / "sample"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / ".index"))
FINANCIALS_CSV = Path(os.getenv("FINANCIALS_CSV", DATA_DIR / "financials.csv"))

EMBEDDINGS = os.getenv("EMBEDDINGS", "hashing")  # hashing | sentence-transformers
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER = os.getenv("RERANKER", "bm25")  # bm25 | cross-encoder
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")  # mock | groq | openai | gemini
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
