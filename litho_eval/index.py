"""Chunk indexes and retrievers shared by every RAG arm."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

from bench.draft_cases import Passage, split_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_ROOT = REPO_ROOT / ".cache" / "litho_eval" / "indexes"
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    doc: str = ""
    section: str = ""
    char_start: int = 0
    char_end: int = 0
    score: float = 0.0


class DenseIndex:
    def __init__(
        self,
        chunks: Sequence[Passage],
        embedding_model: str,
        persist_dir: Path,
    ) -> None:
        from langchain_community.vectorstores import Chroma
        from langchain_core.documents import Document
        from langchain_ollama import OllamaEmbeddings

        self.chunks = list(chunks)
        self.embedding_model = embedding_model
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        chunk_digest = hashlib.sha256(
            "".join(chunk.text_hash for chunk in self.chunks).encode("utf-8")
        ).hexdigest()[:12]
        self.collection_name = "litho_" + chunk_digest
        embeddings = OllamaEmbeddings(model=embedding_model)

        def open_store() -> "Chroma":
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=str(self.persist_dir),
            )

        self.db = open_store()
        if self.db._collection.count() != len(self.chunks):
            try:
                self.db.delete_collection()
            except Exception:
                pass
            self.db = open_store()
            documents = [
                Document(
                    page_content=chunk.text,
                    metadata={
                        "id": chunk.text_hash,
                        "doc": chunk.doc,
                        "section": chunk.section,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    },
                )
                for chunk in self.chunks
            ]
            for start in range(0, len(documents), 256):
                self.db.add_documents(documents[start:start + 256])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        pairs = self.db.similarity_search_with_score(query, k=k)
        results = []
        for document, score in pairs:
            metadata = document.metadata
            results.append(
                RetrievedChunk(
                    id=str(metadata.get("id", "")),
                    text=document.page_content,
                    doc=str(metadata.get("doc", "")),
                    section=str(metadata.get("section", "")),
                    char_start=int(metadata.get("char_start", 0)),
                    char_end=int(metadata.get("char_end", 0)),
                    score=float(score),
                )
            )
        return results


class BM25Index:
    def __init__(self, chunks: Sequence[Passage]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        self.corpus_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k]
        return [
            RetrievedChunk(
                id=self.chunks[index].text_hash,
                text=self.chunks[index].text,
                doc=self.chunks[index].doc,
                section=self.chunks[index].section,
                char_start=self.chunks[index].char_start,
                char_end=self.chunks[index].char_end,
                score=float(scores[index]),
            )
            for index in order
        ]


class HybridRetriever:
    """Reciprocal rank fusion of dense and BM25 candidates."""

    def __init__(
        self,
        dense: DenseIndex | None,
        bm25: BM25Index | None,
        top_k: int = 7,
        candidate_k: int = 25,
        rrf_k: int = RRF_K,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        limit = k or self.top_k
        ranked = []
        if self.dense is not None:
            ranked.append(self.dense.search(query, self.candidate_k))
        if self.bm25 is not None:
            ranked.append(self.bm25.search(query, self.candidate_k))

        scores: dict[str, float] = defaultdict(float)
        by_id: dict[str, RetrievedChunk] = {}
        for results in ranked:
            for rank, chunk in enumerate(results, start=1):
                scores[chunk.id] += 1.0 / (self.rrf_k + rank)
                by_id.setdefault(chunk.id, chunk)

        ordered = sorted(by_id.values(), key=lambda chunk: (-scores[chunk.id], chunk.id))[:limit]
        return [replace(chunk, score=scores[chunk.id]) for chunk in ordered]


class RerankRetriever:
    """FlashRank rerank over hybrid candidates."""

    def __init__(
        self,
        inner: HybridRetriever,
        top_k: int = 7,
        candidate_k: int = 25,
        reranker: Callable[[str, list[RetrievedChunk]], list[RetrievedChunk]] | None = None,
    ) -> None:
        self.inner = inner
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.reranker = reranker
        self._compressor = None

    def _get_compressor(self):
        if self._compressor is None:
            from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

            self._compressor = FlashrankRerank(
                model="ms-marco-TinyBERT-L-2-v2",
                top_n=self.top_k,
            )
        return self._compressor

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        limit = k or self.top_k
        candidates = self.inner.search(query, self.candidate_k)
        if self.reranker is not None:
            return self.reranker(query, candidates)[:limit]

        from langchain_core.documents import Document

        compressor = self._get_compressor()
        documents = [
            Document(
                page_content=chunk.text,
                metadata={
                    "id": chunk.id,
                    "doc": chunk.doc,
                    "section": chunk.section,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            )
            for chunk in candidates
        ]
        compressed = compressor.compress_documents(documents, query)
        by_id = {chunk.id: chunk for chunk in candidates}
        results = []
        for document in compressed:
            chunk = by_id.get(str(document.metadata.get("id")))
            if chunk is not None:
                results.append(chunk)
        return results[:limit]


class IndexBundle:
    def __init__(
        self,
        chunks: Sequence[Passage],
        cache_root: Path,
        embedding_model: str,
    ) -> None:
        self.chunks = list(chunks)
        self.cache_root = Path(cache_root)
        self.embedding_model = embedding_model
        self._dense: DenseIndex | None = None
        self._bm25: BM25Index | None = None

    @property
    def dense(self) -> DenseIndex:
        if self._dense is None:
            slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", self.embedding_model)
            self._dense = DenseIndex(
                self.chunks,
                self.embedding_model,
                self.cache_root / f"chroma_{slug}",
            )
        return self._dense

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            self._bm25 = BM25Index(self.chunks)
        return self._bm25


_INDEX_CACHE: dict[tuple[str, int, int, str], IndexBundle] = {}


def get_index(
    corpus_path: str | Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embedding_model: str = "bge-m3:latest",
    cache_root: str | Path = DEFAULT_INDEX_ROOT,
) -> IndexBundle:
    path = Path(corpus_path)
    text = path.read_text(encoding="utf-8", errors="replace")
    key = (
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        chunk_size,
        chunk_overlap,
        embedding_model,
    )
    if key not in _INDEX_CACHE:
        _INDEX_CACHE[key] = IndexBundle(
            split_corpus(text, chunk_size, chunk_overlap),
            Path(cache_root),
            embedding_model,
        )
    return _INDEX_CACHE[key]
