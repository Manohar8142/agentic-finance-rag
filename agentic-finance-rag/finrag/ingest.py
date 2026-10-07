"""Load documents, split them into overlapping chunks, embed them and build the vector index."""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class Chunk:
    text: str
    source: str
    section: str = ""
    chunk_id: str = ""
    meta: dict = field(default_factory=dict)


def load_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    if suffix == ".csv":
        return csv_to_text(path)
    raise ValueError(f"unsupported file type: {path.name}")


def csv_to_text(path: Path) -> str:
    """Turn a financial table into sentences so the numbers are retrievable as text too."""
    lines = [f"# Quarterly financials table ({path.name}), USD millions"]
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            period = row.pop("period")
            facts = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in row.items())
            lines.append(f"In {period}: {facts}.")
    return "\n\n".join(lines)


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown on headings, keeping each section's heading as metadata."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def chunk_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Pack paragraphs (then sentences) into chunks of about ``size`` chars with ``overlap``."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+", para) if s)
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            cur = f"{tail} {u}".strip()
        else:
            cur = f"{cur} {u}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def chunk_documents(paths: list[Path]) -> list[Chunk]:
    out: list[Chunk] = []
    for path in paths:
        text = load_text(path)
        title = next((ln.lstrip("#").strip() for ln in text.splitlines() if ln.startswith("# ")), path.stem)
        for section, body in split_sections(text):
            if body.startswith(">") and len(body) < 200:
                continue  # skip the short disclaimer banner under each title
            for i, piece in enumerate(chunk_text(body)):
                out.append(Chunk(text=piece, source=path.name, section=section, chunk_id=f"{path.stem}:{len(out)}:{i}", meta={"title": title}))
    return out


def discover(data_dir: Path = config.DATA_DIR) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".md", ".txt", ".pdf", ".csv"} and p.name.lower() != "readme.md")


def build_index(paths: list[Path] | None = None, index_dir: Path = config.INDEX_DIR):
    from .vectorstore import FaissStore
    from .embeddings import get_embedder

    chunks = chunk_documents(paths or discover())
    store = FaissStore(get_embedder())
    store.add(chunks)
    store.save(index_dir)
    return store


if __name__ == "__main__":
    s = build_index()
    print(f"Indexed {len(s.chunks)} chunks into {config.INDEX_DIR}")
