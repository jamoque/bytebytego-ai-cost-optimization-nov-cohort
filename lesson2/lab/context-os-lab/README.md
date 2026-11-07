# Context OS Lab

Build a **Context OS** from scratch — overlapping chunking, cosine RAG
retrieval, a TTL'd LRU semantic cache, extractive compression, and
conversation compaction — until it reproduces the four-arm cost experiment
from the parent lab.

The hands-on pipeline (RAG over Chroma, generation *and* semantic-cache
hit/miss evaluation inside the Lab 1 LiteLLM mesh + Qdrant) lives in
[`../rag-lab/`](../rag-lab/). Work through that first if you have not watched
the gateway decide hit vs miss and fall through to RAG on a miss. **This
exercise needs no API key, no model download, and no network** — it rebuilds
the same engines offline, the way Lab 1's prompt-cache exercise did.

## The big idea

Every token you send to a model is a token you pay for. Three independently
useful levers cut that bill:

```
query ─► semantic cache ─► RAG top-k ─► compress chunks ─► compact history ─► LLM
              │ hit
              └── reuse the stored answer (0 generation tokens)
```

- **RAG retrieval** sends the matching slices of a knowledge base instead of
  the whole corpus.
- **Semantic caching** skips generation entirely when a new question means
  the same thing as a previous one.
- **Compression** drops irrelevant sentences from retrieved text.
  **Compaction** bounds a growing conversation so older turns don't get
  re-billed forever.

A cache **hit** costs $0 on the generation meter. A miss pays for whatever
survived retrieval + compression + compaction — not the raw corpus.

## Setup

```bash
cd lesson2/lab/context-os-lab
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt    # just pytest

# Fill in the TODOs in context_os.py, then:
pytest -v                          # your definition of "done"
python run_demo.py --no-sleep      # replays the 4-arm run through your engine
```

A correct implementation prints a stuffing arm that re-sends the whole corpus
every time, a RAG arm that cuts prompt tokens to the top-k chunks, a cache
arm that turns paraphrases into hits, and a full pipeline whose conversation
prompt stays near the compaction budget. The method contracts are in
`context_os.py`.

## Layout

```
context-os-lab/
├── README.md              # you are here
├── context_os.py          # <- start here (stubs to implement)
├── test_context_os.py     # test suite
├── reference_solution.py  # complete solution (don't peek early)
├── mock_model.py          # fake "expensive" model for latency wins
├── run_demo.py            # replays the parent lab's 4-arm experiment
├── fixtures/              # tiny corpus, queries, conversation
└── requirements.txt
```

## Prerequisites

- Python 3.10+

## References

- Lecture demo: [`../../lecture/semantic-caching-demo/`](../../lecture/semantic-caching-demo/)
- Hands-on lab: [`../rag-lab/`](../rag-lab/)
- Hands-on overview: [`../exercise_overview.md`](../exercise_overview.md)
