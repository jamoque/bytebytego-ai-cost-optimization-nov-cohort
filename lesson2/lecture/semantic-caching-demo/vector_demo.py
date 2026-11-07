#!/usr/bin/env python3
"""
Semantic vector-database demo.

A hands-on tour of the primitives behind semantic caching:

  1. embed        turn a sentence into an embedding vector and inspect it
  2. seed         write a set of illustrative sentences to a vector database
  3. list         show what's stored
  4. compare      print a cosine-similarity matrix over the seeded sentences
  5. search       embed a NEW sentence and find its nearest stored neighbors
  6. interactive  a small REPL that ties it all together

Everything runs locally. Embeddings come from a small sentence-transformer
model (all-MiniLM-L6-v2, 384 dimensions); storage + nearest-neighbor search
come from a persistent Chroma collection on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "sentences.json"
DB_DIR = HERE / ".chroma"
COLLECTION_NAME = "sentences"
MODEL_NAME = "all-MiniLM-L6-v2"


def log(msg: str = "") -> None:
    """Print immediately (model load + first embed can take a moment)."""
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Lazy singletons: importing/loading these is slow, so defer until first use.  #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log("Install dependencies first: pip install -r requirements.txt")
        sys.exit(1)
    log(f"Loading embedding model '{MODEL_NAME}' (first run downloads ~90 MB)...")
    return SentenceTransformer(MODEL_NAME)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed and L2-normalize so dot product == cosine similarity."""
    model = get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_client():
    try:
        import chromadb
    except ImportError:
        log("Install dependencies first: pip install -r requirements.txt")
        sys.exit(1)
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(reset: bool = False):
    client = get_client()
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    # Cosine space => Chroma distance = 1 - cosine_similarity.
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_fixture_sentences() -> list[dict[str, Any]]:
    data = json.loads(FIXTURES.read_text())
    return data["sentences"]


# --------------------------------------------------------------------------- #
# Commands                                                                     #
# --------------------------------------------------------------------------- #
def cmd_embed(args: argparse.Namespace) -> None:
    text = args.text
    vector = embed([text])[0]
    dim = len(vector)
    norm = sum(v * v for v in vector) ** 0.5
    preview_n = min(args.preview, dim)

    log()
    log(f'Sentence : "{text}"')
    log(f"Dimensions: {dim}")
    log(f"L2 norm   : {norm:.4f} (normalized -> ~1.0)")
    log(f"Vector values:")
    head = ", ".join(f"{v:+.4f}" for v in vector[:(preview_n//2)])
    tail = ", ".join(f"{v:+.4f}" for v in vector[(dim-preview_n//2):])
    log(f"  [{head}, ..., {tail}]")

    if args.full:
        log()
        log("Full vector:")
        log(json.dumps([round(v, 6) for v in vector]))


def cmd_seed(args: argparse.Namespace) -> None:
    sentences = load_fixture_sentences()
    collection = get_collection(reset=not args.append)

    ids = [s["id"] for s in sentences]
    docs = [s["text"] for s in sentences]
    metadatas = [{"topic": s.get("topic", "")} for s in sentences]

    log(f"Embedding and writing {len(docs)} sentences to the vector DB...")
    vectors = embed(docs)
    collection.upsert(
        ids=ids, documents=docs, embeddings=vectors, metadatas=metadatas
    )

    log(f"Stored {collection.count()} vectors in collection '{COLLECTION_NAME}'.")
    log(f"Persisted at: {DB_DIR}")
    log()
    log("Contents:")
    for s in sentences:
        log(f"  [{s['id']:<9}] ({s.get('topic', ''):<9}) {s['text']}")


def cmd_list(args: argparse.Namespace) -> None:
    collection = get_collection()
    if collection.count() == 0:
        log("Vector DB is empty. Run:  python vector_demo.py seed")
        return
    got = collection.get(include=["documents", "metadatas"])
    rows = sorted(
        zip(got["ids"], got["documents"], got["metadatas"]),
        key=lambda r: (r[2] or {}).get("topic", ""),
    )
    log(f"{collection.count()} vectors in '{COLLECTION_NAME}':")
    for _id, doc, meta in rows:
        topic = (meta or {}).get("topic", "")
        log(f"  [{_id:<9}] ({topic:<9}) {doc}")


def cmd_compare(args: argparse.Namespace) -> None:
    sentences = load_fixture_sentences()
    ids = [s["id"] for s in sentences]
    docs = [s["text"] for s in sentences]
    vectors = embed(docs)

    log()
    log("Cosine similarity matrix (1.00 = identical meaning, 0.00 = unrelated):")
    log()
    header = "           " + " ".join(f"{i:>9}" for i in ids)
    log(header)
    for i, row_id in enumerate(ids):
        cells = []
        for j in range(len(ids)):
            sim = cosine(vectors[i], vectors[j])
            cells.append(f"{sim:>9.2f}")
        log(f"{row_id:<11}" + " ".join(cells))

    log()
    log("Nearest neighbor for each sentence (excluding itself):")
    for i, s in enumerate(sentences):
        best_j, best_sim = -1, -1.0
        for j in range(len(sentences)):
            if j == i:
                continue
            sim = cosine(vectors[i], vectors[j])
            if sim > best_sim:
                best_sim, best_j = sim, j
        match = sentences[best_j]
        flag = "  <- same topic" if match.get("topic") == s.get("topic") else ""
        log(
            f'  "{s["text"]}"'
        )
        log(
            f"      -> {best_sim:.2f}  \"{match['text']}\"{flag}"
        )


def cmd_search(args: argparse.Namespace) -> None:
    collection = get_collection()
    if collection.count() == 0:
        log("Vector DB is empty. Seed it first:  python vector_demo.py seed")
        return

    query = args.text
    k = min(args.k, collection.count())
    query_vec = embed([query])[0]

    result = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    docs = result["documents"][0]
    metas = result["metadatas"][0]
    dists = result["distances"][0]
    ids = result["ids"][0]

    log()
    log(f'Query: "{query}"')
    log(f"Top {k} nearest neighbors in the vector DB:")
    log()
    log(f"  {'rank':<5}{'similarity':<12}{'id':<11}{'topic':<11}sentence")
    log("  " + "-" * 70)
    for rank, (_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, dists), start=1
    ):
        sim = 1.0 - dist  # cosine space
        topic = (meta or {}).get("topic", "")
        log(f"  {rank:<5}{sim:<12.3f}{_id:<11}{topic:<11}{doc}")

    top_sim = 1.0 - dists[0]
    log()
    if top_sim >= args.threshold:
        log(
            f"CACHE HIT  (top similarity {top_sim:.3f} >= threshold {args.threshold:.2f})"
        )
        log(f'  Reuse the stored answer for: "{docs[0]}"')
    else:
        log(
            f"CACHE MISS (top similarity {top_sim:.3f} <  threshold {args.threshold:.2f})"
        )
        log("  No semantically close entry — compute a fresh answer and store it.")


def cmd_interactive(args: argparse.Namespace) -> None:
    collection = get_collection()
    if collection.count() == 0:
        log("Seeding the vector DB for you...")
        cmd_seed(argparse.Namespace(append=False))
        collection = get_collection()

    log()
    log("Interactive semantic search. Type a sentence to search the vector DB.")
    log("Commands: :list  :quit")
    log(f"Cache-hit threshold: {args.threshold:.2f}  |  top-k: {args.k}")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            log()
            break
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            break
        if line == ":list":
            cmd_list(argparse.Namespace())
            continue
        cmd_search(
            argparse.Namespace(text=line, k=args.k, threshold=args.threshold)
        )
    log("Bye.")


def cmd_reset(args: argparse.Namespace) -> None:
    get_collection(reset=True)
    log(f"Cleared collection '{COLLECTION_NAME}'.")


# --------------------------------------------------------------------------- #
# Argument parsing                                                             #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_embed = sub.add_parser("embed", help="Embed a sentence and inspect the vector")
    p_embed.add_argument("text", help="Sentence to embed")
    p_embed.add_argument(
        "--preview", type=int, default=8, help="How many leading values to show"
    )
    p_embed.add_argument(
        "--full", action="store_true", help="Print the entire vector as JSON"
    )
    p_embed.set_defaults(func=cmd_embed)

    p_seed = sub.add_parser("seed", help="Write illustrative sentences to the DB")
    p_seed.add_argument(
        "--append",
        action="store_true",
        help="Add to existing DB instead of resetting it first",
    )
    p_seed.set_defaults(func=cmd_seed)

    p_list = sub.add_parser("list", help="List what's stored in the DB")
    p_list.set_defaults(func=cmd_list)

    p_compare = sub.add_parser(
        "compare", help="Cosine-similarity matrix over the illustrative sentences"
    )
    p_compare.set_defaults(func=cmd_compare)

    p_search = sub.add_parser(
        "search", help="Similarity-search a new sentence against the DB"
    )
    p_search.add_argument("text", help="Query sentence")
    p_search.add_argument("--k", type=int, default=3, help="Neighbors to return")
    p_search.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Cosine similarity at/above which a query counts as a cache hit",
    )
    p_search.set_defaults(func=cmd_search)

    p_repl = sub.add_parser("interactive", help="Interactive search REPL")
    p_repl.add_argument("--k", type=int, default=3)
    p_repl.add_argument("--threshold", type=float, default=0.55)
    p_repl.set_defaults(func=cmd_interactive)

    p_reset = sub.add_parser("reset", help="Delete all stored vectors")
    p_reset.set_defaults(func=cmd_reset)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
