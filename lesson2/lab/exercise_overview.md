# Lab 2: RAG Retrieval + Semantic Caching + Context Compression

**Course:** ByteByteGo AI Cost Optimization - Nov Cohort | **Instructor**: Jeremy Hintz
**Estimated time to complete:** 90-120 minutes

## 1. Purpose

In lecture we learned several techniques for avoiding sending tokens into the model by retaining information from past model invocations. The most effective cost-saving techniques fight the opposite instinct: when in doubt, stuff more context into the prompt (e.g. a whole knowledge base, an entire conversation, complete tool result, etc) and hope the model ignores what it doesn't need. 

- **Semantic caching**: use a prior response when a new question *means the same thing* as a previous one
- **Compression + compaction**: shrink retrieved text for *this* query, and bound a growing context window so older turns aren't re-billed forever
- **RAG retrieval of memory**: send the matching *slices* of a knowledge base, not the whole corpus


In this lab you will stand up a working pipeline in front of a small customer-support knowledge base (**Northstar Commerce**), then measure each lever independently and stacked. **Every query still enters the LiteLLM mesh you built in Lab 1.** The gateway embeds the incoming request, searches a vector DB, and decides cache hit vs miss — the same diagram as lecture. A hit is served from the gateway's response cache and never reaches OpenAI/Anthropic. A miss falls through to the provider. RAG / compression / compaction are assembled by the Python client into the messages the gateway evaluates — so hit vs miss is always a gateway decision, even when the miss-path workflow ran locally first.

You'll finish by implementing the three engines from the ground up (that half is offline, like Lab 1's prompt-cache exercise).

## 2. Learning Objectives

By the end of this lab, you will be able to:

1. **Explain** why stuffing a corpus into the prompt is a cost bug, not a retrieval strategy
2. **Index** a knowledge base into a local vector store and retrieve top-k chunks for a query
3. **Measure** the token delta between stuffing and RAG, and how that delta moves as `k` changes
4. **Turn on semantic caching inside the LLM gateway**, read hit vs miss from the gateway's response headers, and tune `similarity_threshold` where the decision actually lives
5. **Bound** prompt size with extractive compression (on retrieved text) and compaction (on conversation history)
6. **Compare** four arms of a cost experiment using mesh-reported token usage and recommend a production default for this workload class

## 3. Architecture

```
user query
    │
    ▼
LLM Gateway (Lab 1 LiteLLM mesh)
    │
    ├─ Embed (query → vector) ──► Vector DB (Qdrant) ── similarity search
    │                                              │
    │                    similarity ≥ threshold?   │
    │                         │                    │
    │                        YES ──► Response cache ──► return cached answer
    │                         │
    │                         NO
    ▼
  workflow (this lab's Python client, then back through the same gateway)
    RAG retrieve → compress chunks → compact history → provider LLM
```

Every query hits the gateway. Hit vs miss is **not** decided in `pipeline.py`. The client always calls `chat.completions`; LiteLLM embeds the messages, searches Qdrant, and either returns a cached completion or forwards to the provider. You read the decision from the `x-litellm-semantic-similarity` response header.

LiteLLM keys the semantic cache on the **entire messages array**. That is why this lab disables gateway caching on multi-turn conversations (consecutive turns look almost identical and would false-hit). Single-shot Q&A is the right workload for this cache — LiteLLM's own docs say the same thing.

`prompt_tokens` / `completion_tokens` come from the provider via the mesh. Retrieved vs compressed counts still use a local word splitter so you can see what RAG and compression did before the tokenizer.

## 4. Prerequisites

Before starting the timed portion, make sure you have:

- **Completed Lab 1** and can still start the LiteLLM proxy from `lesson1/lab/llm-mesh-lab` (same venv, same provider keys, same master key `sk-mesh-lab-1234`).
- **Docker**, so you can run a local Qdrant as the gateway's vector DB (`docker run` one container).
- **Provider API keys exported** (`OPENAI_API_KEY` required for `mesh-gpt-mini` and for the `mesh-embed` embedding model the cache uses; `ANTHROPIC_API_KEY` if you want `mesh-claude-haiku`).
- **Installed Python 3.10+** and the ability to create a virtual environment.
- **Optionally completed the lecture demo** at `lesson2/lecture/semantic-caching-demo/`. If you already downloaded `all-MiniLM-L6-v2` there, RAG retrieval reuses it. If not, the first embed downloads ~90 MB.
- **Langfuse is optional but useful.** If you wired `langfuse_otel` in Lab 1, copy that callback into `mesh_config.yaml`. LiteLLM tags traces with `cache_hit` — gateway traffic and provider traffic are no longer the same number.

⚠️ **Spend:** well under **\$1** if you keep the Lab 1 small-model aliases. Do not point the mesh at a frontier model for this lab. Arm A (stuffing) is the expensive arm; that is the point.

> Retrieved/compressed counts use a local word splitter. Billed `prompt_tokens` / `completion_tokens` come from the provider through the mesh. They will not match exactly — different tokenizers, plus the system prompt the pipeline adds. Treat the splitter as a *relative* RAG meter and the mesh usage as the bill.

## 5. A Note on Using AI

Like many (most) things these days, using AI (Claude Code, Cursor, etc) will probably knock this exercise out at break-neck speed. That may or may not be the best way to actually learn something, depending on the person. If you want to use AI to write the code for this lab, go for it! But I encourage you to reason about the code itself and try to understand the ideas. This isn't for a grade or anything, it's purely for you to learn.

## Part 1: Index, Retrieve, and Call the Mesh (~25 min)

### 1.1 Set up the project

```bash
cd lesson2/lab/rag-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You can reuse the lecture demo's venv if you prefer (`pip install pyyaml openai` if that venv doesn't have them).

### 1.2 Start Qdrant and the Lab 1 mesh (with semantic cache)

The gateway needs a vector DB for hit/miss evaluation. In a **second terminal**:

```bash
docker run -d --name lab2-qdrant -p 6333:6333 qdrant/qdrant
```

Then start LiteLLM on the **Lab 2** mesh config — it is your Lab 1 config plus an embedding alias and Qdrant semantic cache. Reuse last week's venv:

```bash
cd lesson1/lab/llm-mesh-lab
source .venv/bin/activate
export OPENAI_API_KEY="sk-..."          # if not already in this shell
export ANTHROPIC_API_KEY="sk-ant-..."   # optional; this lab defaults to mesh-gpt-mini
export QDRANT_API_BASE=http://localhost:6333
export QDRANT_API_KEY=unused            # local Qdrant ignores this; some LiteLLM versions want the var
litellm --config ../../lesson2/lab/rag-lab/mesh_config.yaml
```

You should see the proxy on http://0.0.0.0:4000. Verify the aliases:

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-mesh-lab-1234"
```

**Checkpoint 0:** the list includes `mesh-gpt-mini` **and** `mesh-embed`. Open `mesh_config.yaml` and find `similarity_threshold` — that number is the hit/miss decision, and it lives in the gateway, not in the Python client.

Leave this terminal running. Client `config.yaml` already points at `http://localhost:4000/v1` with `api_key: sk-mesh-lab-1234` and `model: mesh-gpt-mini`. Swap the model string to `mesh-claude-haiku` if you want the other Lab 1 provider — one string, same as last week.

### 1.3 Inspect the knowledge base

Eight policy documents live in `knowledge_base/`. They are written so that a refund question should *not* need the SLA, the API rate-limit guide, or the data-retention schedule — but a stuffing baseline will send all of them anyway.

```bash
python -c "
from pipeline import load_documents, token_count, corpus_token_count
for d in load_documents():
    print(f'{d[\"source\"]:<22} {token_count(d[\"text\"]):>5} tokens')
print(f'{\"TOTAL\":<22} {corpus_token_count():>5} tokens')
"
```

Skim `knowledge_base/refunds.txt` and `knowledge_base/sla.txt`. One of those is the right context for "How do I get a refund for a damaged order?" The other is expensive noise.

### 1.4 Index the corpus

A starter `config.yaml` is already in this directory. Confirm the retriever block looks like:

```yaml
retriever:
  chunk_tokens: 80
  chunk_overlap: 20
  k: 3
  min_score: 0.25
  stuff: false
```

Then build the vector index:

```bash
python index_kb.py
```

You should see ~8 documents split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and written to `.chroma/`. Re-run `index_kb.py` whenever you change `chunk_tokens` or `chunk_overlap`; retrieval settings (`k`, `min_score`, `stuff`) take effect without re-indexing.

### 1.5 Retrieve, then ask the mesh

A starter `client_test.py` is already in this directory:

```python
from pipeline import Pipeline, token_count

QUESTION = "How do I get a refund for a damaged order?"

pipeline = Pipeline.from_config("config.yaml")
```

Write code to retrieve chunks for `QUESTION` and print, for each hit: source filename, cosine score, token count, and a short preview of the text. Then print two totals: `pipeline.corpus_token_count()` (the stuffing baseline) and the sum of retrieved chunk tokens.

Then call `pipeline.ask(QUESTION)` and print the model answer, `llm_calls`, `cache_hit`, and `semantic_similarity`. That `ask` is a real mesh round-trip: the OpenAI SDK, your Lab 1 master key, the `mesh-gpt-mini` alias. The gateway evaluates the request for a cache hit *before* calling the provider.

**Checkpoint 1:** retrieved sources should be `refunds.txt` / `returns.txt` (not `sla.txt` or `rate_limits.txt`), retrieved tokens should be a small fraction of the corpus total, and `ask` should return a refund-policy answer with `llm_calls=1` and `cache_hit=False` (cold cache). The LiteLLM terminal should show the request.

**Don't be fooled if a shipping chunk sneaks into the top-3.** "Damaged order" co-occurs with replacement shipping in the corpus, and cosine retrieval is not a keyword filter. That is the retrieval-quality vs. cost tradeoff you are about to measure: raising `k` recovers recall and also sends more tokens.

**Think:** Trace one retrieved chunk back to the paragraph it came from in `knowledge_base/`. Why did overlapping windows (`chunk_overlap: 20`) make that paragraph more likely to be found as a self-contained chunk? What goes wrong if you set overlap to 0 and a sentence straddles a chunk boundary?

## Part 2: Semantic Cache Inside the Gateway (~20 min)

In lecture, two paraphrases of "how do I get a refund?" landed close together in vector space and scored a **CACHE HIT**. A trivia question did not. That decision now lives **inside the gateway**, matching the diagram: the mesh embeds the incoming messages, searches Qdrant, and either returns the response cache or falls through to the provider.

To keep the cache key equal to the *query* (not RAG-expanded context), turn retrieval off for this part so the gateway sees a stable system prompt plus the user question.

In `config.yaml`:

```yaml
retriever:
  enabled: false
```

### 2.1 Confirm the miss path (cache bypassed)

Leave `semantic_cache.enabled: false` (the starter default). That flag does **not** turn the gateway cache off globally — it tells this client to send `cache.no-cache` and `cache.no-store` on every request, so the gateway still *sees* the query but does not read or write Qdrant. In a Python shell or a throwaway script:

```python
from pipeline import Pipeline

pipe = Pipeline.from_config("config.yaml")
queries = [
    "How do I get a refund for my order?",
    "What's the process to return an item and get my money back?",
    "What's the capital of France?",
]
for q in queries:
    r = pipe.ask(q)
    print(f"hit={r.cache_hit}  sim={r.semantic_similarity}  llm={r.llm_calls}  {q}")
    print(f"  {r.answer[:160]}")
```

All three should be misses (`hit=False`, `llm=1`) — three gateway requests **and** three provider calls. Watch the LiteLLM terminal: you should see the completion forwarded to OpenAI/Anthropic each time.

### 2.2 Let the gateway cache

In client `config.yaml`:

```yaml
semantic_cache:
  enabled: true
```

Do **not** change the Python. The client will stop sending `no-cache` / `no-store`; the gateway's `similarity_threshold: 0.80` in `mesh_config.yaml` is now in charge. Re-run the same three questions.


**Checkpoint 2:** all three questions still appear in the LiteLLM request log (they entered the gateway). Question 1 is a miss (`llm=1`, `cache_hit=False`) and a store into Qdrant. Question 2 (the paraphrase) is a **hit**: `cache_hit=True`, `llm=0`, `semantic_similarity` above threshold, and the **same answer string** as question 1. The LiteLLM log should show the request was served from cache (look for `x-litellm-semantic-similarity` / no provider round-trip). Question 3 is still a miss.

If question 2 misses, lower `similarity_threshold` in `mesh_config.yaml` (try `0.70`) and **restart LiteLLM** — the threshold is gateway config, so the client cannot override it. If question 3 hits, raise the threshold.

**Think:** Trace the hops for question 2: client → gateway → embed → Qdrant similarity search → response cache → return. The provider was never called. Your gateway log is now *complete* relative to user traffic, but provider spend is not. If Langfuse is on, find the `cache_hit` tag. Why is that a better observability story than a client-side cache that never talks to the mesh?

**Think:** LiteLLM embeds the **entire messages array**, system prompt included. Why did we turn `retriever.enabled` off for this part? What would happen to paraphrase hits if the retrieved chunks differed between question 1 and question 2?

**Think:** The cache key is whatever the gateway received, TTL is 300s in `mesh_config.yaml`. If an operator updates `refunds.txt`, the gateway will keep serving the old answer until TTL expires. What TTL would you pick for a support corpus that changes weekly vs. a product catalog that changes hourly?

### 2.3 (Optional) Turn RAG back on and watch the cache key change

Set `retriever.enabled: true` and re-run the two refund paraphrases. The gateway now embeds *query + retrieved context*. If the top-k chunks are identical, you may still hit; if they differ, a paraphrase that hit in 2.2 can miss. That is the cost of putting the workflow *before* the cache key instead of *on the miss path only*.

## Part 3: Compress Retrieved Text, Compact History (~25 min)

Retrieval caps *which documents* enter the prompt. Compression and compaction cap *how much of them, and of the conversation, survive*.

- **Compression** is per-request and query-aware: drop sentences in the retrieved text that do not overlap the question.
- **Compaction** is per-conversation and query-*unaware*: when history exceeds a token budget, keep the system context plus the newest turns, and replace dropped turns with a marker.

They are easy to conflate because both make prompts smaller. They fail in opposite ways. Compression can drop the one sentence that actually answered the question. Compaction can drop the turn where the customer said the order number.

### 3.1 Watch a conversation grow

Leave compression and compaction **disabled**. Run:

```python
from pipeline import Pipeline, Message, token_count
import json

pipe = Pipeline.from_config("config.yaml")
turns = json.loads(open("fixtures/conversation.json").read())
history = []
print(f"{'turn':>4} {'prompt':>7} {'retr':>6}  question")
for i, text in enumerate(turns, start=1):
    r = pipe.ask(text, history=history)
    print(f"{i:>4} {r.prompt_tokens:>7} {r.retrieved_tokens:>6}  {text[:60]}")
    history.append(Message("user", text))
    history.append(Message("assistant", r.answer))
```

The conversation starts as a damaged-blender refund, then pivots to a password reset, a failed charge, an SLA credit, and an API 429. Without compaction, turn 10 still carries turn 1's pitcher-crack narrative in the prompt — and the mesh bills you for it. Each turn should show up as a separate generation in the LiteLLM log.

Set `retriever.enabled: true` again before this part. Leave `semantic_cache.enabled` however you like: the client **will not** ask the gateway to cache once `history` is non-empty. LiteLLM's semantic cache is for single-shot prompts; consecutive conversation turns are nearly identical embeddings and would replay the previous answer. (Read the warning in their docs if you want the scar tissue without earning it.)

### 3.2 Enable both levers

In `config.yaml`:

```yaml
compression:
  enabled: true
  max_tokens: 120

compaction:
  enabled: true
  token_budget: 400
```

Re-run the conversation loop (new `Pipeline()`, empty `history`).

**Checkpoint 3:** retrieved tokens still vary by question (RAG is doing its job), and mesh `prompt_tokens` stop climbing — turn 10 is no longer billed for the whole thread. That 400 figure is the local word-splitter budget, not the provider tokenizer, so mesh `prompt_tokens` level off *near* 400 (expect something in the mid-400s) rather than landing on it. Confirm the budget itself with `sum(token_count(m.content) for m in r.prompt_messages)`, which stays at or under 400 on later turns. Scan a late-turn prompt (`r.prompt_messages`) and find a `[compacted: N earlier turns]` marker. On this corpus the marker shows up around turn 7, not on the password-reset turn.

**Question:** Compare the password-reset turn in both runs, then the first turn that contains a `[compacted: …]` marker. Retrieval for that password question should already be `password_reset.txt`, so compression is shrinking *those* sentences — the blender story was never in the retrieved chunks. Compaction is what drops the blender turns, and only once history no longer fits. Which of those two would still be wrong if the customer said "the order number is on the first message in this chat"?

**Think:** `token_budget: 400` is smaller than some single retrieved-chunk concatenations. If the system context plus the current question already overflow the budget, compaction will drop history to make room — including history you might actually need. What would you change first: raise the budget, lower `k`, or compress harder?

## Part 4: The Cost Experiment (~20 min)

You now have the three levers. Act like a cost engineer: measure them independently and stacked.

### 4.1 Run the four-arm experiment

`run_experiment.py` ignores the `enabled` flags you were toggling and builds four clean arms from the same config numbers (`k`, threshold, budgets):

| Arm | Retrieval | Semantic cache | Compress | Compact |
| --- | --------- | -------------- | -------- | ------- |
| A   | stuff entire corpus | off | off | off |
| B   | top-k RAG | off | off | off |
| C   | top-k RAG | on | off | off |
| D   | top-k RAG | on | on | on |

```bash
python run_experiment.py
```

There are two workloads. **Queries** is a short Q&A list with paraphrases (isolates the gateway cache — expect hits on the three paraphrase pairs: same number of *gateway* requests as questions, fewer *provider* calls). The billing question and the SLA question retrieve the same chunks, so in arm C that pair can also clear 0.80 even though the questions are different. Read the SLA answer before counting that hit as a win. Arm D compresses the chunks first, which changes the embedded text; that same pair often falls back under the threshold.

**Conversation** is the 10-turn support thread. The client bypasses the semantic cache once `history` is non-empty, so turns 2–10 of arm C match arm B and all reach the provider. Turn 1 has empty history, so it is still eligible for the cache and can hit a refund answer stored during the query workload. Do not expect arm C's conversation provider-call count to be 10.

If you have Langfuse from Lab 1, filter to tag `lab2` and session `lab2-arm-C` vs `lab2-arm-A`. Arm C's query session should show `cache_hit` on the paraphrases. Arm A should be the expensive outlier on input tokens.

To start from a clean Qdrant collection between full reruns, delete it and **restart LiteLLM**. The proxy creates the collection at startup; a delete while it is running makes later cache writes 404 and every paraphrase misses.

```bash
curl -X DELETE http://localhost:6333/collections/lab2_semantic_cache
# then restart: litellm --config lesson2/lab/rag-lab/mesh_config.yaml
```

### 4.2 Build a comparison table

From the two summaries, fill in:

| | A stuffing | B RAG only | C RAG + cache | D full pipeline |
| --- | ---: | ---: | ---: | ---: |
| Query workload: gateway requests |  |  |  |  |
| Query workload: provider calls |  |  |  |  |
| Query workload: cache hits |  |  |  |  |
| Query workload: total prompt tokens (mesh) |  |  |  |  |
| Query workload: estimated cost |  |  |  |  |
| Conversation: total prompt tokens (mesh) |  |  |  |  |
| Conversation: estimated cost |  |  |  |  |
| Conversation: mean prompt tokens / turn |  |  |  |  |

### 4.3 Food for Thought Questions

**Question 1:** Identical questions rarely produce identical retrieved-token counts across arms B and D. Where did the difference come from — retrieval (`k`) or compression? Which one is more under your control for a single query, and how?

**Question 2:** Based only on your table, what would be your default pipeline for this workload class (customer-support Q&A over a slow-changing policy corpus, paraphrases common, threads 5–15 turns)? What measurements does this experiment *not* capture that could reverse your recommendation? Open one stuffing answer and one RAG answer for the same question — cheaper is not a win if the model no longer has the policy it needs.

**Question 3:** Arm C's cache sits *inside* the mesh. Suppose 40% of production traffic is paraphrase hits. Which number in your table scales with that hit rate — gateway requests, or provider calls? If you alert on "LiteLLM request count" instead of "provider tokens," what incident will you miss, and what incident will you over-page?

**Question 4:** Point at one concrete prompt-token number in arm D's conversation table and walk it back: was the saving from not stuffing, from a cache hit, from dropping sentences, or from dropping turns? If you can only keep one lever in production next quarter, which one, and what in your data says so?

---

## Advanced Exercise: Build a Context OS From Scratch

During lecture and Parts 1–4 you ran a black-boxed pipeline in front of your Lab 1 mesh: you toggled YAML flags and watched mesh usage move. You never had to write cosine top-k, a sliding TTL, or the compaction suffix rule.

This exercise goes inside that black box: you implement the engines themselves and make them reproduce the four-arm accounting you just observed — no mesh, no Chroma, no network, no spend. The embedder is a hashed bag-of-words stand-in for MiniLM, so paraphrases that *share wording* land close together. The mechanics (threshold, TTL, LRU, budgets) are the same ones a production system runs on top of real embeddings.

### Learning objectives

By the end you will be able to explain, from first principles:

1. Why RAG chunking uses overlapping windows, and why a sentence that straddles a boundary can vanish from retrieval.
2. How cosine top-k plus a minimum score is both a quality filter and a cost cap.
3. Why a semantic cache is nearest-neighbor over *queries*, why the TTL slides on a hit, and why LRU matters once the cache is bounded.
4. Why compression is query-aware sentence selection and compaction is a query-unaware suffix over a token budget — and why those two can drop different things.

### The mental model

- Documents are split into overlapping **chunks**. Retrieval embeds the query, scores every chunk, and returns the top-k above `min_score`.
- The **semantic cache** embeds the query and hits if the closest unexpired stored query is within `threshold`. Hits skip retrieval and generation. A hit refreshes the TTL; overflow evicts LRU.
- **Compression** keeps the sentences in retrieved text that overlap the query, up to `max_tokens`.
- **Compaction** keeps all system messages plus the newest suffix of the conversation that fits in `budget`, and records how many turns were dropped.

### Your task

1. Navigate to https://github.com/jamoque/bytebytego-ai-cost-optimization-nov-cohort
2. Clone the repository (if you haven't already)
3. Edit only `lesson2/lab/context-os-lab/context_os.py`. Implement the six methods marked TODO, in order:

| Function | Purpose |
| --- | --- |
| `chunk_text` | Overlapping token windows over a document |
| `Retriever.retrieve` | Cosine top-k with a minimum score |
| `SemanticCache.lookup` | Nearest cached query, threshold + sliding TTL |
| `SemanticCache.store` | Write an entry; LRU-evict under `max_entries` |
| `compress` | Extractive sentence selection for a query |
| `compact` | Bound a conversation to a token budget |

Everything else is provided: the tokenizer, lexical embedder, dataclasses, and the `Pipeline` that wires the pieces. Each function has a detailed docstring spelling out the contract.

### Workflow

```bash
cd lesson2/lab/context-os-lab
python -m venv .venv && source .venv/bin/activate   # or reuse the parent venv
pip install -r requirements.txt

# 1. Implement the TODOs in context_os.py, checking your work as you go:
pytest -v

# 2. Once green, watch your engine reproduce the four-arm run:
python run_demo.py --no-sleep
```

A correct implementation prints a stuffing arm that re-sends the whole corpus every time, a RAG arm that cuts prompt tokens to the top-k chunks, a cache arm that turns paraphrases into hits, and a full pipeline whose conversation prompt stays near the compaction budget — the same shape you saw against the live mesh.

### Definition of done

1. All tests in `test_context_os.py` pass (`pytest -v`).
2. `python run_demo.py --no-sleep` shows arm A as the expensive baseline, arm C with cache hits on paraphrases, and arm D bounding conversation prompt tokens.
3. You can explain why a cache hit on turn 2 of a paraphrase pair costs *zero* generation tokens, and why turn 1 of that pair cannot.

If you run out of gas or just want to run a completed solution, you can find the solution in `reference_solution.py`.

### References

- Lecture semantic-caching demo: [`../lecture/semantic-caching-demo/README.md`](../lecture/semantic-caching-demo/README.md)
- Hands-on pipeline: [`rag-lab/pipeline.py`](rag-lab/pipeline.py)
- Anthropic prompt caching (the *other* cache, from Lab 1 — prefix-based, not semantic): https://platform.claude.com/docs/en/build-with-claude/prompt-caching

## References

- Lab 1 mesh (required): [`../../lesson1/lab/exercise_overview.md`](../../lesson1/lab/exercise_overview.md)
- Lecture demo (embed, compare, search, hit/miss): [`../lecture/semantic-caching-demo/README.md`](../lecture/semantic-caching-demo/README.md)
- Chroma cosine collections: https://docs.trychroma.com/docs/collections/configure
- Sentence-Transformers `all-MiniLM-L6-v2`: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- LiteLLM Proxy docs: https://docs.litellm.ai/docs/simple_proxy
- LiteLLM semantic caching (Qdrant / Redis / Valkey): https://docs.litellm.ai/docs/proxy/caching_semantic
- Langfuse ↔ LiteLLM (if you left it on from Lab 1): https://langfuse.com/integrations/gateways/litellm
