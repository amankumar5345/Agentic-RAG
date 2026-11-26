"""Build the separate litho_injected Chroma collection.

The real index is never read, deleted or upserted by this module.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from bench.injection.cases import CASES_PATH, InjectionCase, read_injection_cases

COLLECTION_NAME = "litho_injected"
DEFAULT_PERSIST = Path(".cache/litho_eval/injection")
DEFAULT_EMBEDDING_MODEL = "bge-m3:latest"


def build_injected(
    cases: Sequence[InjectionCase],
    persist_dir: str | Path = DEFAULT_PERSIST,
    embed_texts: Callable[[list[str]], list[list[float]]] | None = None,
    collection_name: str = COLLECTION_NAME,
):
    import chromadb
    from langchain_ollama import OllamaEmbeddings

    directory = Path(persist_dir)
    directory.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(directory))
    collection = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [case.injected_passage for case in cases]
    if embed_texts is None:
        embedder = OllamaEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
        embed_texts = embedder.embed_documents
    vectors = embed_texts(texts)

    collection.upsert(
        ids=[case.id for case in cases],
        documents=texts,
        embeddings=vectors,
        metadatas=[
            {
                "attack_type": case.attack_type,
                "doc": case.doc,
                "section": case.section,
                "char_start": case.char_start,
                "char_end": case.char_end,
            }
            for case in cases
        ],
    )
    return collection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build litho_injected")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST))
    args = parser.parse_args(argv)

    cases = read_injection_cases(args.cases)
    if not cases:
        raise SystemExit("no injection cases; run python -m bench.injection.cases")
    collection = build_injected(cases, args.persist_dir)
    print(f"litho_injected: {collection.count()} passages at {args.persist_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
