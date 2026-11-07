# Semantic Caching Demo

Instead of caching on an exact string match, **semantic caching** caches on
*meaning*. Each request is turned into an embedding vector; a new request that
is semantically close to a previous one (measured by vector similarity) reuses
the stored answer.

A traditional cache only hits when the key is byte-for-byte identical. "How do I
get a refund?" and "How can I return this and get my money back?" would be two
separate misses. A semantic cache embeds both into vectors, notices they point
in nearly the same direction, and serves one answer for both.

This is a command-line tour of those primitives: turn text into embedding
vectors, store them in a vector database, and find the nearest neighbor for a
brand-new sentence. If a new query lands close enough to a stored one, that's a
**cache hit** — you can reuse the previous answer instead of recomputing it.

Everything runs locally:

- **Embeddings** — `all-MiniLM-L6-v2` (sentence-transformers), 384 dimensions, L2-normalized so dot product equals cosine similarity.
- **Vector store** — a persistent [Chroma](https://www.trychroma.com/) collection on disk (`.chroma/`), using cosine distance.

## Setup

```bash
cd lesson2/lecture/semantic-caching-demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> First run downloads the local embedding model (~90 MB). No API key needed.

## The five steps

### 1. Embed a sentence and inspect the vector

```bash
python vector_demo.py embed "How do I get a refund?"
```

Shows the dimensionality, the L2 norm (~1.0 after normalization), and a preview of
the raw values. Add `--full` to dump the entire 384-dim vector as JSON, or
`--preview 16` to see more leading values.

### 2. See how similar sentences compare

```bash
python vector_demo.py compare
```

Prints a cosine-similarity matrix over the bundled illustrative sentences
(`fixtures/sentences.json`) and each sentence's nearest neighbor. Paraphrases of the
same intent score high (~0.6–0.9); unrelated sentences score near 0.

### 3. Write the sentences to the vector database

```bash
python vector_demo.py seed        # resets, then writes all fixtures
python vector_demo.py list        # inspect what's stored
```

Use `--append` to keep existing entries.

### 4. Similarity-search a new sentence

```bash
python vector_demo.py search "I need my money back for an order"
```

Embeds the new sentence, returns the top-k nearest stored neighbors with their
similarity scores, and labels the result a **CACHE HIT** or **CACHE MISS** based on
`--threshold` (default `0.55`).

```bash
python vector_demo.py search "when is my package showing up?" --k 5 --threshold 0.6
```

> The `all-MiniLM-L6-v2` model scores genuine paraphrases around 0.6–0.8 and
> unrelated sentences below ~0.4, so `0.55` cleanly separates hits from misses.
> Tune `--threshold` to trade off cache hit rate against the risk of a wrong reuse.

### 5. Do it interactively

```bash
python vector_demo.py interactive
```

Seeds the DB if empty, then drops you into a REPL: type any sentence to search.
`:list` shows the DB, `:quit` exits.

## Suggested demo flow

```bash
python vector_demo.py embed "How do I get a refund?"   # a vector is just numbers
python vector_demo.py compare                          # some vectors are close
python vector_demo.py seed                              # store the close ones
python vector_demo.py search "how can I return this and be reimbursed?"   # HIT
python vector_demo.py search "what's the capital of France?"              # MISS
```

The refund paraphrase hits the `refunds` cluster; the unrelated trivia question
finds nothing close and misses — exactly the decision a semantic cache makes.

## Command reference

| Command | Purpose |
|---------|---------|
| `embed <text> [--preview N] [--full]` | Embed one sentence, inspect the vector |
| `compare` | Cosine-similarity matrix over the fixtures |
| `seed [--append]` | Write illustrative sentences to the DB |
| `list` | Show stored vectors |
| `search <text> [--k N] [--threshold T]` | Nearest-neighbor search + hit/miss verdict |
| `interactive [--k N] [--threshold T]` | Search REPL |
| `reset` | Delete all stored vectors |

## Files

- `vector_demo.py` — the CLI (all commands)
- `fixtures/sentences.json` — illustrative sentences, grouped into semantic clusters
- `requirements.txt` — `chromadb`, `sentence-transformers`
- `.chroma/` — the on-disk vector DB (created on first `seed`; git-ignored)
