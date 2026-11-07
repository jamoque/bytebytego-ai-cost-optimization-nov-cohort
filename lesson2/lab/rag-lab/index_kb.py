#!/usr/bin/env python3
"""
Chunk the Northstar knowledge base, embed each chunk, and write them to Chroma.

    python index_kb.py           # reset + re-index (default)
    python index_kb.py --append  # add to whatever is already stored
"""

from __future__ import annotations

import argparse

from pipeline import (
    chunk_document,
    embed,
    get_collection,
    load_config,
    load_documents,
    log,
    token_count,
    corpus_token_count,
    DB_DIR,
    COLLECTION_NAME,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing collection instead of resetting it",
    )
    args = parser.parse_args()

    cfg = load_config()
    chunk_tokens = int(cfg["retriever"]["chunk_tokens"])
    overlap = int(cfg["retriever"]["chunk_overlap"])

    docs = load_documents()
    collection = get_collection(reset=not args.append)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for doc in docs:
        pieces = chunk_document(doc["text"], chunk_tokens, overlap)
        for i, piece in enumerate(pieces):
            ids.append(f"{doc['source']}::{i}")
            documents.append(piece)
            metadatas.append(
                {
                    "source": doc["source"],
                    "chunk_index": i,
                    "tokens": token_count(piece),
                }
            )

    log(f"Knowledge base: {len(docs)} documents, {corpus_token_count()} tokens")
    log(
        f"Chunking at {chunk_tokens} tokens (overlap {overlap}) → {len(documents)} chunks"
    )
    log("Embedding and writing to the vector DB...")
    vectors = embed(documents)
    collection.upsert(
        ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas
    )
    log(f"Stored {collection.count()} vectors in collection '{COLLECTION_NAME}'.")
    log(f"Persisted at: {DB_DIR}")
    log()
    log("Chunks per source:")
    counts: dict[str, int] = {}
    for meta in metadatas:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    for source, n in sorted(counts.items()):
        log(f"  {source:<22} {n} chunks")


if __name__ == "__main__":
    main()
