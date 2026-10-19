"""Stage 2: handbook retrieval with ChromaDB.

Embeddings are computed here and handed to Chroma explicitly, so the store
works the same with the offline hashing embedder and with a real model.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb

from .schemas import ParsedRequest, Passage


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Bag-of-words feature hashing. No downloads, deterministic, weak semantics:
    good enough to exercise the pipeline, not to measure retrieval quality."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                h = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "little")
                vec[h % self.dim] += 1.0 if (h >> 63) == 0 else -1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


class ChromaDefaultEmbedder:
    """Chroma's bundled all-MiniLM-L6-v2 (ONNX); downloads the model on first use."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(list(texts))]


def make_embedder(name: str) -> Embedder:
    return ChromaDefaultEmbedder() if name == "default" else HashingEmbedder()


@dataclass
class Chunk:
    id: str
    source: str
    text: str


def chunk_text(text: str, source: str, max_chars: int = 900, overlap: int = 150) -> list[Chunk]:
    """Split on blank lines, then pack paragraphs into ~max_chars windows."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for p in paragraphs:
        if current and len(current) + len(p) + 2 > max_chars:
            chunks.append(current)
            current = current[-overlap:] + "\n\n" + p if overlap else p
        else:
            current = f"{current}\n\n{p}" if current else p
        while len(current) > max_chars:  # a single very long paragraph
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap :]
    if current:
        chunks.append(current)
    stem = Path(source).stem
    return [Chunk(id=f"{stem}#{i}", source=source, text=c) for i, c in enumerate(chunks)]


def read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    text = path.read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)  # drop authoring notes in Markdown


def iter_documents(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in {".md", ".txt", ".pdf"} and path.is_file():
            yield path


class HandbookStore:
    def __init__(self, embedder: Embedder, path: Path | None = None, collection: str = "handbooks"):
        self.embedder = embedder
        client = chromadb.PersistentClient(path=str(path)) if path else chromadb.EphemeralClient()
        self.collection = client.get_or_create_collection(
            collection, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def __len__(self) -> int:
        return self.collection.count()

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
            metadatas=[{"source": c.source} for c in chunks],
        )

    def ingest_folder(self, folder: Path) -> int:
        count = 0
        for path in iter_documents(folder):
            chunks = chunk_text(read_document(path), source=path.name)
            self.add(chunks)
            count += len(chunks)
        return count

    def search(self, query: str, k: int = 4) -> list[Passage]:
        if len(self) == 0:
            return []
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=min(k, len(self)),
            include=["documents", "metadatas", "distances"],
        )
        return [
            Passage(id=pid, source=meta["source"], text=doc, score=round(1.0 - dist, 4))
            for pid, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]


def query_for(parsed: ParsedRequest) -> str:
    """What to look up in the handbooks for this request."""
    parts = [parsed.category, parsed.summary, *parsed.required_skills, *parsed.required_certifications]
    return " ".join(p for p in parts if p)
