# Lab 1: LLM Mesh \+ Observability \+ Prompt Caching

**Course:** ByteByteGo AI Cost Optimization \- Nov Cohort \| **Instructor**: Jeremy Hintz  
**Estimated time to complete:** 90-120 minutes

## 1\. Purpose

In lecture we saw that agentic LLM workloads re-purchase their context on every step, and that costs amplify through retries, fan-out, and tool-schema bloat. None of that can be *managed* until it can be *observed*. Observation requires every application to talk to every provider directly through its own SDK, its own keys, and its own (sometimes nonexistent and fragmented) logging.

An **LLM Mesh** (also called an LLM gateway or AI gateway) solves this by putting a single control plane between your applications and your model providers:

- **One API surface**: every app speaks the same language to the gateway, regardless of which provider actually serves the request  
- **Centralized key and access management**: provider keys live in one place  
- **Routing and aliasing**: "cheap fast model" can become a name the mesh resolves independently, making it easier to swap providers later on (we will look at this piece much more deeply later in the course)  
- **A single choke point for observability**: every token flows through one place where it can be traced, metered, and logged

In this lab you will stand up a working mesh in front of **two providers**, then instrument it with **Langfuse** so that every request produces raw events, metrics, logs, and traces you can analyze. 

You'll finish by setting up prompt caching in your API requests.

## 2\. Learning Objectives

By the end of this lab, you will be able to:

1. **Explain** the role of an LLM mesh/gateway in a cost-optimization strategy  
2. **Deploy** LiteLLM Proxy locally and register two independent providers behind one endpoint  
3. **Route** identical client code to different providers purely through mesh configuration  
4. **Instrument** the mesh with Langfuse and identify where each of the four observability pillars lives: raw events, metrics, logs, and traces  
5. **Analyze** real telemetry (cost per request, token counts, latency) across two providers  
6. **Optimize** by adding prompt caching to provider API requests

   

    

## 3\. Architecture

![][image1]

## 4\. Prerequisites

Before starting the timed portion, make sure you have:

- **Installed Python 3.10+** and the ability to create a virtual environment.  
- **Obtain 2+ provider API keys.** The reference materials and configurations in this lab will use **OpenAI** and **Anthropic**. If you don't have paid accounts or prefer other providers, please go ahead\! LiteLLM supports all of them, you'll just adjust the provider prefix in the config.   
  - How-To Guides:  
    - Anthropic → [https\://platform.claude.com/docs/en/get-api-key](https://platform.claude.com/docs/en/get-api-key)   
    - OpenAI → [https\://developers.openai.com/api/docs/quickstart](https://developers.openai.com/api/docs/quickstart)   
  - ⚠️ **Note**: **Under \$1** expected spend if you use paid keys and small models  
  - ⚠️ **Important Reminder:** provider keys should be stored as as local environment variables. Please please please. It’s easy to accidentally and commit / push an API key. Don’t do it\!  
- **Create a free Langfuse Cloud account**: sign up at https\://cloud.langfuse.com (Hobby tier is sufficient). Self-hosting Langfuse via Docker is also great\!

> 

## 5\. A Note on Using AI 

Like many (most) things these days, using AI (Claude Code, Cursor, etc) will probably knock this exercise out at break-neck speed. That may or may not be the best way to actually learn something, depending on the person. If you want to use AI to write the code for this lab, go for it\! But I encourage you to reason about the code itself and try to understand the ideas. This isn’t for a grade or anything, it’s purely for you to learn.

## Part 1: Install and Configure the Mesh (\~25 min)

### 1.1 Set up the project and install LiteLLM

This directory already exists in the cloned repo:

```bash
cd lesson1/lab/llm-mesh-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install 'litellm[proxy]'
pip install openai
```

### 1.2 Export your provider keys

| \$ export OPENAI\_API\_KEY="sk-..." \$ export ANTHROPIC\_API\_KEY="sk-ant-..." |
| :---- |

⚠️ Reminder: DO NOT COMMIT THESE\! :)

### 1.3 Write the mesh configuration

A starter `config.yaml` is already in this directory. Confirm it looks like:

| model\_list:   \# \---- Provider A: OpenAI \----   \- model\_name: mesh-gpt-mini                     \# your own alias; feel free to change     litellm\_params:       model: openai/gpt-4o-mini                   \# model to proxy; feel free to change       api\_key: os.environ/OPENAI\_API\_KEY   \# \---- Provider B: Anthropic \----   \- model\_name: mesh-claude-haiku     litellm\_params:       model: anthropic/claude-haiku-4-5-20251001       api\_key: os.environ/ANTHROPIC\_API\_KEY general\_settings:   master\_key: sk-mesh-lab-1234                    \# client \<\> mesh auth  |
| :---- |

### 

### 1.4 Start the mesh

| \$ litellm \--config config.yaml |
| :---- |

You should see the proxy start on http\://0.0.0.0:4000.

### 1.5 Verify

In a second terminal window, ask the mesh what models it exposes:

| \$ curl http\://localhost:4000/v1/models \-H "Authorization: Bearer sk-mesh-lab-1234" |
| :---- |

**Checkpoint 1:** the response should list your two aliases (mesh-gpt-mini, mesh-claude-haiku) and *only* aliases, no provider model IDs.

**Don't be fooled if you see "owned\_by": "openai" on both entries.** The list endpoint reveals both are following OpenAI-*protocol* facade: LiteLLM hardcodes `owned_by` to `"openai"` and stamps every entry with the same placeholder created timestamp (1677610602), purely for compatibility with clients that expect OpenAI's response shape. Said otherwise, provider identity is deliberately not part of the client-facing contract.

If you want to confirm:

| \$ curl http\://localhost:4000/v1/models/mesh-claude-haiku \-H "Authorization: Bearer sk-mesh-lab-1234" \# → "owned\_by": "anthropic" |
| :---- |

## Part 2: Route Traffic Through the Mesh (\~15 min)

### 2.1 Call Provider A with curl

| curl http\://localhost:4000/v1/chat/completions \\   \-H "Authorization: Bearer sk-mesh-lab-1234" \\   \-H "Content-Type: application/json" \\   \-d '{     "model": "mesh-gpt-mini",     "messages": \[{"role": "user", "content": "In one sentence, what is a KV cache?"}\]   }' |
| :---- |

### 2.2 Call Provider B by changing one string

Re-run the same command, swapping the model param

### 2.3 Do it from Python with the plain OpenAI SDK

A starter `client_test.py` is already in this directory:

| from openai import OpenAI QUESTION \= "What is the capital of France?" \# Note: this is the STANDARD OpenAI Protocol SDK client \= OpenAI(     base\_url="http\://localhost:4000/v1",     api\_key="sk-mesh-lab-1234", ) |
| :---- |

Write code to loop through the model aliases and print the answer to QUESTION for each.

**✅ Checkpoint 2:** both aliases respond through the unmodified OpenAI SDK.

**🤔Think:** Your client passed `api_key="sk-mesh-lab-1234"` and it worked for both OpenAI and an Anthropic model. Trace the two authentication hops that actually occurred. Why might splitting auth this way be beneficial for e.g. (a) offboarding an employee (b) capping a team's monthly spend?

## Part 3: Wire Up Observability with Langfuse (\~30 min)

### 3.1 Get Langfuse credentials

1. Log in to Langfuse Cloud and create a project (e.g., llm-mesh-lab).  
2. In **Project Settings → API Keys**, create a key pair. You'll get a **public key** (pk-lf-...) and a **secret key** (sk-lf-...).

### 3.2 Export them and point at your data region

| \$ export LANGFUSE\_PUBLIC\_KEY="pk-lf-..." \$ export LANGFUSE\_SECRET\_KEY="sk-lf-..." \$ export LANGFUSE\_OTEL\_HOST="https\://us.cloud.langfuse.com"       \# assuming US region  |
| :---- |

### 3.3 Enable the callback in the mesh config

Add to `config.yaml`:

| litellm\_settings:   callbacks: \["langfuse\_otel"\] |
| :---- |

This uses LiteLLM's OpenTelemetry-based Langfuse integration: every request through the mesh is exported as an OTel trace to Langfuse, carrying the request/response payloads, token usage, timing, and any metadata you attach. 

Note what we did **not** do: you did not touch client\_test.py. Instrumentation at the mesh means every current and future client is observed for free.

### 3.4 Restart the mesh and generate labeled traffic

Stop the proxy (Ctrl+C) and start it again with the same command. Then create [`workload.py`](http://workload.py):

| import time from openai import OpenAI client \= OpenAI(base\_url="http\://localhost:4000/v1", api\_key="sk-mesh-lab-1234") PROMPTS \= \[     "Summarize in 2 sentences why prefix caching lowers LLM inference cost.",     "Explain the difference between prefill and decode in one short paragraph.",     "List 3 reasons an agent's context grows across steps. Be brief.",     "What is speculative decoding? Answer in under 50 words.",     "Why is output pricing usually higher than input pricing? 2 sentences.", \] for alias in \["mesh-gpt-mini", "mesh-claude-haiku"\]:     for i, prompt in enumerate(PROMPTS):         t0 \= time.time()         resp \= client.chat.completions.create(             model=alias,             messages=\[{"role": "user", "content": prompt}\],             \# Metadata flows through the mesh into Langfuse:             extra\_body={                 "metadata": {                     "generation\_name": f"lab4-q{i}",                     "session\_id": "lab1-cost-experiment",                     "tags": \["lab1", alias\],                 }             },         )         print(f"{alias} q{i}: {time.time()-t0:.2f}s, "               f"{resp.usage.prompt\_tokens}+{resp.usage.completion\_tokens} tokens")  |
| :---- |

### 

Run it: `python workload.py` (10 requests total; give Langfuse \~30–60 seconds to ingest).

### 3.5 Tour the four pillars in the Langfuse UI

Open your project and find each of the following. This mapping is the conceptual core of the lab — every observability stack you'll ever meet (Datadog, Grafana, OTel collectors) has some version of these four:

|  | Where you'll find it | What to look at |
| :---- | :---- | :---- |
| **Traces** | **Tracing → Traces** | One trace per request. Open one: the tree of observations, the generation span, its duration. In a multi-step agent, one trace would contain *many* generations — here each trace has one. |
| **Raw events** | Inside a trace → click the **generation** | The raw ingested record: exact input messages, exact model output, model name, parameters, token usage. This is the ground-truth event from which everything else is derived. |
| **Metrics** | **Dashboards**  (and the trace table columns) | Aggregates over events: total cost, cost per model, token counts, latency percentiles, request volume over time. |
| **Logs** | The **Traces table**  (filterable, chronological request record)  | Langfuse's trace list *is* your request log: filter by tag `lab4`, by model, by error status. The proxy's stdout gives you the infrastructure-level log line per request. |

**✅ Checkpoint 3:** Screenshot (a) the trace list filtered to tag `lab4`, (b) one opened trace showing a generation with input/output and token usage, and (c) a dashboard view showing **cost broken down by model**.

**📝 Question 3:** Your `workload.py` sent `session_id` and `tags` through the mesh. Explain how you would use exactly these two fields to answer, six months from now: *"Which product feature is responsible for 80% of our LLM spend?"* What would you need to enforce at the mesh for that query to stay answerable?

**📝 Question 4:** Distinguish a *trace* from a *raw event* from a *metric* using your own screenshots as evidence: point to one concrete number in your dashboard and walk it back to the raw event(s) it was computed from.

---

## Part 4: The Cost Experiment (\~20 min)

You now have 5 identical prompts answered by two providers, with full telemetry. Act like a cost engineer.

### 4.1 Build a comparison table

From Langfuse (trace table \+ generation details), fill in:

|  | Provider A (mesh-gpt-mini) | Provider B (mesh-claude-haiku) |
| :---- | :---- | :---- |
| Total cost (5 requests) |  |  |
| Avg input tokens |  |  |
| Avg output tokens |  |  |
| Avg latency |  |  |
| Cost per 1K output tokens (effective) |  |  |

### 4.2 Food for Thought Questions

**🤔 Question 1:** Identical prompts rarely produce identical token counts across providers. Where did the difference come from in your data, input side (tokenization) or output side (verbosity)? Which one is more under your control, and how?

**🤔 Question 2:** Based only on your table above, what would be your routing recommendation for this workload class (i.e. short Q\&A, low stakes). What measurements does your experiment *not* capture that could reverse your recommendation?

## References

- LiteLLM Proxy docs: [https\://docs.litellm.ai/docs/simple\_proxy](https://docs.litellm.ai/docs/simple_proxy)  
- LiteLLM ↔ Langfuse logging: [https\://docs.litellm.ai/docs/proxy/logging](https://docs.litellm.ai/docs/proxy/logging)  
- Langfuse ↔ LiteLLM integration guide: [https\://langfuse.com/integrations/gateways/litellm](https://langfuse.com/integrations/gateways/litellm)  
- Langfuse tracing data model: [https\://langfuse.com/docs/tracing](https://langfuse.com/docs/tracing)

## 

## 

## Advanced Exercise: Build a Prompt Caching Engine From Scratch

During lecture, we ran a black-boxed experiment that showed the power of simple prompt caching for decreasing costs from API providers. We saw Anthropic's API report cache\_creation\_input\_tokens and cache\_read\_input\_tokens, and confirmed that caching a large static prefix slashes cost and latency on later turns.

This exercise goes inside that black-box: you implement the caching engine itself and make it reproduce the create / read / fresh accounting you observed against the real API: no API key, no network, no spend.

### Learning objectives

By the end you will be able to explain, from first principles:

1. Why prompt caching is prefix-based and block-aligned, and why a single changed byte near the start invalidates the cache for everything after it.  
2. How a chained hash lets a server match the longest cached prefix in O(blocks).  
3. Why the cache is ephemeral (TTL) and why reads refresh the TTL.  
4. How a bounded cache uses LRU eviction under memory pressure.

### The mental model

- Caching works in fixed-size **blocks**. Only complete blocks are cacheable; the trailing partial block is re-processed every time.  
- Each block gets a **chained hash**: hash(prev\_hash \+ block\_tokens), so a block only matches when the entire prefix is identical.  
- Per request: find the longest run of leading blocks already cached (**read**/hit), write the new blocks after it (**create**/write), and the leftover tail is **fresh**.  
- Entries **expire** after a TTL; a hit **slides** the TTL forward.  
- When blocks exceed the budget, the **least-recently-used** ones are evicted.

### Your task

1. Navigate to [https://github.com/jamoque/bytebytego-ai-cost-optimization-nov-cohort](https://github.com/jamoque/bytebytego-ai-cost-optimization-nov-cohort)   
2. Clone the repository  
3. Edit only `lesson1/lab/prompt-caching-lab/prompt_cache.py`. Implement the five methods marked TODO, in order:  
   

| Function | Purpose |
| :---- | :---- |
| PrefixCache.\_block\_hashes | Split tokens into complete blocks; assign prefix-dependent chained hashes |
| PrefixCache.lookup | Count contiguous leading blocks that are cached and unexpired |
| PrefixCache.store | Write new blocks; refresh the TTL of hits; trigger eviction |
| PrefixCache.\_evict | Drop least-recently-used blocks until under max\_blocks |
| PrefixCache.process | Tie it together: accounting for read / write / fresh \+ cost |

Everything else is provided: the tokenizer, dataclasses, and pricing constants. Each function has a detailed docstring spelling out the contract.

### Workflow

```bash
cd lesson1/lab/prompt-caching-lab
python -m venv .venv && source .venv/bin/activate   # or reuse the parent venv
pip install -r requirements.txt

# 1. Implement the TODOs in prompt_cache.py, checking your work as you go:
pytest -v

# 2. Once green, watch your engine reproduce the lecture demo's 8-turn run:
python run_demo.py   # add --no-sleep to skip the latency simulation
```

A correct implementation prints a table where turn 1 is a big **write** and turns 2-8 are \~99% **reads**, with a cost reduction in the 70%+ range — the same shape you saw against the live API.

### Definition of done

1. All tests in test\_prompt\_cache.py pass (pytest \-v).  
2. python run\_demo.py shows turn-1 create spike, turns 2–8 dominated by reads.  
3. You can explain why turn 1 with caching is *more* expensive than baseline.

If you run out of gas or just want to run a completed solution, you can find the solution in reference\_solution.py

### References

- Anthropic prompt caching docs: [https\://platform.claude.com/docs/en/build-with-claude/prompt-caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)  
- Demo experiment: [../lecture/prompt-caching-demo/README.md](../lecture/prompt-caching-demo/README.md)

[image1]: <data:image/png;base64,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+3/vvfcq3aSf8aE/IcuWsNrF5Zdf3rZC+/N94MCBtjpTVz9r1qyy/sorryw/u/9z1Mboj++6665rXX311cVOxHz00UflZ3PfffeVv9H/GN1hmCeeeKKsy+0mLc1zymF15syZficADm0nhNXO9u/fX7YPs2fP7jk869c9S+d2++mnnyrf1UbZZ5WFCxeWQXf9+vWVflV0FNaOvrr879089thjrWuvvbb4vHPnztrfqu748ePlZ43TPPLII7W/wXCELFvCagc2f1YU/HR6w5/vdevWFd+vuOIK59f//Z+c369Ynf4n6dcZbdzuuBUufbo+xx+2jvzab+xo79GjR9uGr6O9VkdYTQNhFQij7STWsKrp7jXtfrswmXbv3l22I9amuEcoly5d2rr44ouLz9OnT69Mmz4/8MAD5e90Wt7otLzbblmb5XK/K2TqqG0nGs+cOXOKz3YwyKe6X375pfXuu++2dXenBcMXsmwJqx10mj+/Xkc7/f78U/fG3aBV9D9G+2ynNozVr169ulJvnnrqqbbxusO+6KKLKkdxtdNwuZcaWNH/OFOn+SSsAvnSdtIr8DWZpt322XXbu98uTKY//vijGN+uXbsq9TrVbtw2xqXv7tFL14YNG2r7dy9T03e1gwqq+qxT95288sorleHZ9Lht5A8//NDW3S33339/2R3D5f+t6xBWO6ibP3fFNboetFt/bje/vq4fv99OLJy88847ZZ0/TCs6KtyJNkC3X136kDLNI2EVyJe2Ewur2l5iLf5+3p2/UfKnw5+eG2+8sfj+1VdfOb9qFZe2Wb/ugRwFS/FvbJoxY0YlkNqpeX98nbj92MEeFV2C599gpTOTuqzhjjvuKK6VVX86S4nJEfL3I6x2YPP35ZdfFnccuhuFO992GkP/yuOPP96xX3224PjNN98U19H8/PPPZXeX/1ufTle4wxP3NzqVoY05hI4E61RN3TSnRvNGWAXype3ETqf7++qYiwVwfR61RYsWldNxyy23FPdyhNi7d295ParOBuqIaj9005TOEk6mlStXjmWZ5iRk+RJWO/B3BCruqfNe/br1durCvs+bN8/9ea268fjUXRu4+13FrhHynTt3rvxs/fpPMlixYkXQuGOl+SKsAvnSdhLzZQC2f+60L0t13z1KWq72SKy6m7swXCHLl7Dahf6XpzsXdXTV1M23ThVYvRtE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>