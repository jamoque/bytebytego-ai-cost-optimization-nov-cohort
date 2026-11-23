# Lab 4: Harness Engineering + Open Models + Jev Decision Models

**Course:** ByteByteGo AI Cost Optimization - Nov Cohort | **Instructor**: Jeremy Hintz
**Estimated time to complete:** 90-120 minutes

## 1. Purpose

In Labs 1-3 we cut cost by changing which tokens reach a model (prompt caching, retrieval, compression) and which model receives them (routing). In lecture we saw that most of an agent's bill is decided somewhere else: in the **harness**, which is everything in an agent except the model. The harness decides what goes into the prompt and in what order, what a tool result is allowed to add to the context, when the loop stops, and whether anything checks the answer before it leaves.

This lab covers three ideas:

- **Harness engineering**: the same model, tools, and tickets cost very different amounts per *solved* ticket depending on the harness around them
- **Open models**: run an open-weight model on your own machine, behind the same mesh, and measure what it costs and what it can do
- **Decision models**: models such as Jev return a typed answer with a probability instead of text. Here one works as a cheap gate that decides which drafts deserve more spending

You will run a real support agent for **Northstar Commerce** (the company from Lab 2) and turn its harness switches on one at a time. It is the lecture's harness ledger ([`harness_cost_savings.py`](../lecture/harness_cost_savings.py)) with the mock model and the mock provider replaced by real ones. **Every model call still enters the LiteLLM mesh you built in Lab 1**, including the calls to models on your own machine, and the agent's knowledge-base tool reads the index you built in Lab 2.

You'll finish by building a decision model out of a chat model's first token (that half is offline, like the earlier advanced exercises).

## 2. Learning Objectives

By the end of this lab, you will be able to:

1. **Read** a harness ledger: split one agent run's cost into input at full price, cache writes, cache reads, output, gate, and hardware
2. **Order** a prompt so the provider's prefix cache can hit, and measure what one volatile line at the top costs
3. **Bound** what a tool result adds to the context, and **stop** a loop that has stopped learning anything
4. **Explain** why dollars per solved ticket, not dollars per attempt, is the number to optimize, and re-price a run when a failure costs money
5. **Run** an open model on your own machine behind the mesh, size its context window, and put a price on its seconds
6. **Call** a decision model (yes/no, choice, score), **choose** its threshold from labeled data, and use it to gate an expensive step
7. **Compare** eight arms of a cost experiment and recommend a harness for this workload

## 3. Architecture

```
support ticket
    │
    ▼
harness.py  (the agent loop; this lab's Python client)
    │   switches: stable_prefix · cap_tools · loop_guard · verify · reasoning
    │
    ├─ tools ─► search_kb ───────────────► Lab 2 Chroma index (lesson2/lab/rag-lab/.chroma)
    │           get_order, get_account, … ► fixtures/crm.json
    ▼
LLM Gateway (Lab 1 LiteLLM mesh, localhost:4000)
    │
    ├─ /v1/chat/completions
    │     mesh-gpt-mini ───────► OpenAI
    │     mesh-claude-haiku ───► Anthropic
    │     mesh-local ──────────► Ollama on your machine   (lab4-agent  = qwen3.5:4b)
    │
    └─ /v1/systemone
          mesh-decide ─────────► Ollama on your machine   (lab4-decide = kev:4b)
          mesh-jev (optional) ─► TypeSafe's hosted Jev
```

**One loop, five switches.** `harness.py` is provided and you do not have to implement it. A ticket comes in, the model calls tools until it writes a draft, and the draft becomes the reply. Each switch in `config.yaml` changes one thing the harness does around that loop. The model, the tools, the playbook, and the tickets are identical in every configuration.

**Three kinds of model behind one gateway.** Closed chat models, an open chat model on your machine, and decision models all sit behind the same master key. The client never learns which is which.

**Chat models write. Decision models decide.** You send a decision model a `state` (text or JSON) and a set of typed `questions`. It reads once, generates nothing, and returns a probability for each permitted answer. The API is called System One (`POST /v1/systemone`). **Jev** is TypeSafe AI's hosted decision model. **Kev** is an open-weight decision model (Apache-2.0, built on Qwen3.5) that Ollama serves through the same API. This lab runs Kev so that nobody needs another account, and Part 3.5 shows the one-alias swap to hosted Jev.

| Tier in `config.yaml` | Mesh alias | Model | Price |
| --- | --- | --- | --- |
| `weak` | `mesh-gpt-mini` | gpt-4o-mini | \$0.15 in / \$0.60 out per million tokens. Cached input \$0.075, automatic |
| `mid` (default) | `mesh-claude-haiku` | Claude Haiku 4.5 | \$1.00 in / \$5.00 out. Cache write \$1.25, cache read \$0.10, and only where the request sets breakpoints |
| `local` | `mesh-local` | qwen3.5:4b on your machine | \$0 on the mesh's meter. The ledger charges `local.usd_per_hour` for the seconds it ran |
| decision | `mesh-decide` | kev:4b on your machine | same as `local` |
| decision (optional) | `mesh-jev` | Jev 1.13.0, hosted | \$0.042 per million input tokens, no output charge |

Dollar figures in the ledger come from the mesh (the `x-litellm-response-cost` header, the same number Langfuse shows). The prices in `config.yaml` split that cost into line items.

## 4. Prerequisites

Before starting the timed portion, make sure you have:

- **Completed Labs 1 & 2:** the LiteLLM proxy from `lesson1/lab/llm-mesh-lab` (same venv, same master key `sk-mesh-lab-1234`), and the vector index in `lesson2/lab/rag-lab/.chroma/` built with the default chunking. This lab's `search_kb` tool reads that index. Lab 3 is not required.
- **Provider API keys:** `OPENAI_API_KEY` serves the `weak` tier. `ANTHROPIC_API_KEY` serves the `mid` tier, which is the default agent and the judge. With only an OpenAI key, set `harness.tier: weak` and `judge.tier: weak` in `config.yaml`. We ran arms A-G once that way: \$0.18 in total, and gpt-4o-mini solved fewer tickets, so the numbers in this document will not match yours.
- **LiteLLM 1.110.2 or newer.** Older versions cannot send a decision request to Ollama (the error is `Unknown Decisions provider 'ollama_chat'`). Upgrade the Lab 1 venv before starting the mesh: `pip install -U 'litellm[proxy]'`
- **Ollama 0.40.4 or newer**, with both models downloaded (about 7 GB). Do step 2.1 before the timed portion. Check your machine against the table below first.
- **Installed Python 3.10+** and the ability to create a virtual environment. (This lab was verified on Python 3.14.)
- **Langfuse is optional but useful.** If you wired `langfuse_otel` in Lab 1, uncomment that callback in `mesh_config.yaml`. Every chat call in this lab is tagged `lab4` and named for its purpose (`lab4-agent`, `lab4-check`, `lab4-judge`, `lab4-handoff`).

**Minimum requirements for the open models (Parts 2 and 3):**

| | macOS | Windows | Linux |
| --- | --- | --- | --- |
| OS | macOS 14 (Sonoma) or newer | Windows 10 22H2 or newer | A 64-bit distribution, x86-64 or ARM64 |
| Minimum we suggest | Apple Silicon (any M-series) with 16 GB of memory | 16 GB of RAM, and an NVIDIA or AMD GPU with 6 GB of VRAM | same as Windows |
| Comfortable | 24 GB of memory or more | 8 GB of VRAM or more (12 GB holds both models at once) | same as Windows |
| GPU driver | nothing to install | NVIDIA driver 551.61 or newer. AMD: a ROCm v7 or Vulkan-capable Radeon driver | NVIDIA driver 550 or newer (570 or newer for older cards). AMD: ROCm v7 |
| Without a GPU | Intel Macs run on the CPU only | runs on the CPU, slowly | runs on the CPU, slowly |
| Disk | 7 GB for the two models | 7 GB for the models, plus about 4 GB for Ollama | 7 GB for the two models |

Where these numbers come from: the OS and driver rows are Ollama's published requirements. The memory rows come from what `ollama ps` reported on our machine. The decision model took 4.7 GB. The chat model took 3.7 GB in the build that Windows and Linux download, and between 4.2 and 9.0 GB in the Apple Silicon build, which grows as it caches prompts. The lab never needs both models in the same moment, and Ollama unloads an idle model when the next one does not fit.

We ran this lab on one machine, an Apple M1 Max with 64 GB. We have not run it on Windows, on Linux, on an Intel Mac, or on a machine with 8 GB. Two things we did measure for smaller machines:

- **CPU only.** With the GPU switched off on our machine, a three-turn ticket took 37 to 62 seconds instead of 17, and a gate decision took about four times as long. Everything still worked.
- **A smaller chat model is not the fix.** `qwen3.5:2b` solved 2 of the 12 tickets. If memory is tight on a Mac, keep the 4B model and switch to its smaller build (step 2.1).

If your machine cannot run the models at all, do Part 1, read Parts 2 and 3 with our numbers, and run Part 4 without the local arms (`python run_experiment.py --arms A,B,C,D,E,F`).

⚠️ **Spend:** one full run of `run_experiment.py` cost us about **\$1.00** on Haiku 4.5 (November 2026), and Parts 1-3 add about \$0.50. Budget **\$2-3** for the whole lab. The open models cost nothing at the API.

**Mileage will vary.** Models are not perfectly repeatable even at temperature 0. We ran the Part 4 experiment three times. Total spend was \$1.00, \$1.01, and \$0.98, and no API arm's cost per attempt moved by more than about 7%. Success moved more: arm C solved 7, 8, and 8 tickets of 12, arm E solved 11, 12, and 11, and the open model solved 6, 6, and 8. The cost structure should be stable. Individual answers may not be.

## 5. A Note on Using AI

Like many (most) things these days, using AI (Claude Code, Cursor, etc) will probably knock this exercise out at break-neck speed. That may or may not be the best way to actually learn something, depending on the person. If you want to use AI to write the code for this lab, go for it! But I encourage you to reason about the code itself and try to understand the ideas. This isn't for a grade or anything, it's purely for you to learn.

## Part 1: Harness Engineering (~35 min)

### 1.1 Set up the project and start the Lab 4 mesh

```bash
cd lesson4/lab/harness-lab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

In a **second terminal**, start LiteLLM on the Lab 4 mesh config. It is your Lab 1 config plus an open chat model and a decision model. Reuse the Lab 1 venv:

```bash
cd lesson1/lab/llm-mesh-lab
source .venv/bin/activate
pip install -U 'litellm[proxy]'         # this lab needs 1.110.2 or newer
export OPENAI_API_KEY="sk-..."          # if not already in this shell
export ANTHROPIC_API_KEY="sk-ant-..."
litellm --config ../../../lesson4/lab/harness-lab/mesh_config.yaml
```

You should see the proxy on http://0.0.0.0:4000. Verify the aliases:

```bash
curl http://localhost:4000/v1/models \
  -H "Authorization: Bearer sk-mesh-lab-1234"
```

**Checkpoint 0:** the list includes `mesh-gpt-mini`, `mesh-claude-haiku`, `mesh-local`, and `mesh-decide`. The last two point at Ollama on your machine. The mesh lists them whether or not Ollama is running, and Part 1 does not use them.

Leave this terminal running. The client's `config.yaml` already points at `http://localhost:4000/v1` with `api_key: sk-mesh-lab-1234`.

### 1.2 Run one ticket through a naive harness

Four fixtures define the workload. Skim them before you run anything:

| File | What it is |
| --- | --- |
| `fixtures/playbook.md` | The agent's system prompt: ground rules, counting rules, worked examples. With the tool schemas it comes to about 4,300 tokens |
| `fixtures/tools.json` | Eight tools. `search_kb` searches the Lab 2 index. The rest look up orders, accounts, uptime, API usage, logins, returns, and shipments |
| `fixtures/crm.json` | The records those tools return. They are verbose, the way real API payloads are |
| `fixtures/tickets.json` | Twelve tickets, each with checks on the facts in the reply. A ticket is solved only if every check passes |

Open `config.yaml` and find the `harness:` section. Every switch is off. That is the agent most teams ship first.

A starter `client_test.py` is already in this directory:

```python
from harness import Harness, print_ledger, print_trace

TICKET = (
    "Hi, this is Dana at Acme Outfitters (account ACME-77). Three things. First, on order NS-48213 "
    ...
)

harness = Harness.from_config("config.yaml")
```

Write code to run `TICKET` through the harness, print the ledger and the trace, and print the reply the customer would get. `harness.run(TICKET)` returns a `Run`. `print_ledger(harness, run)` shows what the run cost by line item. `print_trace(run)` shows every model call and every tool call.

```bash
python client_test.py
```

On our run:

```
Harness  tier=mid stable-prefix=off cap-tools=off loop-guard=off verify=off reasoning=off

  This run                  $   0.0442   3 turns, 9.9s, ended by: done, unverified
    Input at full price     $   0.0401   91%  ███████████████████████████
    Cache writes            $   0.0000
    Cache reads             $   0.0000
    Output                  $   0.0041    9%  ██▌

  call purpose   prompt  full price  cache write  cache read  output  reasoning    call $   total $    sec
     1 agent      4,759       4,759            0           0     236          0   0.00594    0.0059    2.6
     2 agent     16,291      16,291            0           0     164          0   0.01711    0.0231    2.1
     3 agent     19,057      19,057            0           0     425          0   0.02118    0.0442    5.3

  tool  get_order({"order_id": "NS-48213"}) -> 1571 tok
  tool  get_account({"account_id": "ACME-77"}) -> 2007 tok
  tool  get_api_usage({"account_id": "ACME-77"}) -> 5782 tok
  tool  get_uptime({"account_id": "ACME-77", "month": "2026-10"}) -> 1189 tok
  tool  search_kb({"query": "restocking fee opened item Pro subscriber"}) -> 680 tok
  tool  search_kb({"query": "service credit uptime percentage tier subscription fee"}) -> 668 tok
  tool  search_kb({"query": "rate limit requests per minute Pro plan"}) -> 680 tok
```

**Checkpoint 1:** three model calls, about \$0.044, and roughly 90% of it is input at full price. The `prompt` column grows on every call because each call re-sends everything before it. The reply should contain three figures: a \$200.00 refund, a \$180.00 service credit, and a limit of 300 requests per minute that the account exceeded by 150.

**Think:** The model needed two numbers from `get_api_usage`. How many tokens did that result add, and how many times were they paid for? The first call cost 4,759 input tokens before the model had read a single record. What is in those tokens, and how much of it changes from one ticket to the next?

### 1.3 Stable prefix

Find `_stamp` and `_system` in `harness.py`. Somebody added a request id and a clock to the first line of the system prompt for debugging. That one line is different on every call, so no prefix ever repeats. Anthropic also caches nothing unless the request marks where the cacheable prefix ends.

`stable_prefix` fixes both. The volatile line moves to the end of the ticket, and the harness sets two cache breakpoints: one after the playbook and tool schemas, one at the end of the conversation so far. Change your run to:

```python
run = harness.run(TICKET, stable_prefix=True)
```

Any switch in `config.yaml` can be overridden for one run this way. Run it **twice**, less than five minutes apart (Anthropic's cache lives for five minutes).

| On our run | Total | Call 1 | Call 2 | Call 3 |
| --- | ---: | --- | --- | --- |
| Naive (1.2) | \$0.0442 | 4,759 at full price | 16,291 at full price | 19,057 at full price |
| Stable prefix, first run | \$0.0295 | writes 4,757 | reads 4,757, writes 11,527 | reads 16,284, writes 2,746 |
| Stable prefix, second run | \$0.0251 | reads 4,301, writes 454 | reads 4,755, writes 11,527 | reads 16,282, writes 2,765 |

**Checkpoint 2:** on the second run, call 1 reads about 4,300 tokens from cache. That is the playbook plus the tool schemas, and it is the only part two different tickets share. Within a run, each call reads everything the previous call wrote. "Input at full price" drops to almost nothing.

The playbook's length is not an accident. Haiku 4.5 will not cache a prefix shorter than 4,096 tokens, with or without breakpoints.

**Break it on purpose.** Set the breakpoints and leave the volatile line on top:

```python
run = harness.run(TICKET, stable_prefix=True, break_cache=True)
```

Every call writes and none reads. On our run this cost \$0.0545, which is 23% **more** than the naive harness, because a cache write is billed at 1.25 times the input price.

**The other provider.** Run the naive and the stable-prefix versions on gpt-4o-mini by adding `tier="weak"`. OpenAI caches a repeated prefix automatically, with no breakpoints, at half price. On our run the naive harness cost \$0.0048 with zero cached tokens (the volatile line defeats automatic caching too), and the stable prefix cost \$0.0035 and then \$0.0033.

**Think:** After this step, "Cache writes" is the largest line in the ledger (73% on our second run). Which tokens are being written, and what would make that line smaller? That is the next step.

### 1.4 Cap what a tool result can add

A tool result is paid for on every later call. `cap_tools` does three things to each result before the model sees it: it keeps only the fields that the tool's schema promises (`returns` in `fixtures/tools.json`), it compresses knowledge-base passages against the query with Lab 2's `compress`, and it cuts whatever is left at `tool_cap_tokens`.

```python
run = harness.run(TICKET, stable_prefix=True, cap_tools=True)
```

| On our run | Uncapped | Capped |
| --- | ---: | ---: |
| `get_api_usage` result | 5,782 tokens | 50 tokens |
| `get_account` result | 2,007 tokens | 99 tokens |
| Prompt on the last call | 19,057 tokens | 7,189 tokens |
| Cost of the run | \$0.0251 | \$0.0080 |

**Checkpoint 3:** the run costs about a fifth of the naive one. Now read the reply carefully. On our run the service credit came back as **\$60.00**. Fifteen percent of \$1,200 is \$180. The capped reply was also shorter (179 output tokens on the last call against 425): the model went straight to its answers without the working it wrote in 1.2.

Your run may get \$180. Part 4 measures this across all twelve tickets, and on every one of our runs the capped arm solved fewer tickets than the uncapped one. Hold that thought until 1.6.

**Think:** This run is cheaper per attempt than any before it. Is it cheaper per *solved* ticket? What would you need to know to answer?

### 1.5 Stop a loop that has stopped learning

One ticket in the workload cannot be answered. The carrier feed for its order returns "refreshing" on every call, with a sync percentage that creeps upward. Add it to `client_test.py`:

```python
STUCK = "Order NS-51777 still hasn't arrived. Where is the package right now?"

run = harness.run(STUCK, stable_prefix=True, cap_tools=True)
```

Then run it again with `loop_guard=True`.

| On our run | Turns | Cost | Seconds | Reply |
| --- | ---: | ---: | ---: | --- |
| No loop guard | 30 | \$0.0337 | 29.1 | none: cut off at the hard limit |
| Loop guard | 4 | \$0.0042 | 3.9 | says tracking is unavailable and hands off to a specialist |

**Checkpoint 4:** without the guard the agent calls `get_shipment` 30 times and the customer gets nothing. With it, the trace marks the second and third results `[nothing new: 1.00 similar to an earlier result]`, and a fourth call writes a reply. In the naive harness the same ticket cost \$0.21 (you will see it in Part 4, arm A).

The guard has three rules, all in `config.yaml`: a turn budget (`max_turns`), a dollar budget (`max_usd`), and convergence. Convergence embeds each tool result with Lab 2's embedder and compares it with earlier results from the same tool. After `repeat_limit` results in a row at or above `converge_score`, the harness stops the tool loop and asks for a final reply.

**Think:** An exact-match check on the tool result would never have fired here. Why not? The turn budget alone would have stopped this run at 12 turns. What does the convergence rule buy on top of it? Read `hand_off` in `harness.py`: the tool schemas stay in the final request and `tool_choice` forbids using them. What would dropping the schemas have cost?

### 1.6 Check the draft before it leaves

In 1.4 the reply got cheaper and, on our run, wrong. `verify="always"` adds a closing pass: before a draft is sent, the agent is asked to re-derive every figure from the records in the conversation, show its working, and write the final reply.

```python
run = harness.run(TICKET, stable_prefix=True, cap_tools=True, loop_guard=True, verify="always")
print(run.drafts[0])     # the draft before the closing pass
print(run.answer)        # what the customer gets
```

**Checkpoint 5:** the trace shows a fourth call with the purpose `check`. On our run the draft said \$60.00 and the final reply said \$180.00. The run cost \$0.0128 against \$0.0080 without the pass, and took 13.9 seconds against 6.8. The closing pass wrote 743 output tokens, so output is now the largest line in the ledger (52%).

**Optional: buy some thinking.** Add `reasoning="on"` (every turn may think) or `reasoning="sandwich"` (only the opening turn and the closing pass). On our run both cost more and neither changed an answer:

| On our run | Cost | Seconds | Reasoning tokens |
| --- | ---: | ---: | --- |
| Closing pass, no thinking | \$0.0128 | 13.9 | 0 |
| `reasoning="on"` | \$0.0217 | 23.3 | 348 on call 1, 693 on the check |
| `reasoning="sandwich"` | \$0.0224 | 23.4 | 346 on call 1, 616 on the check |

**Think:** With `reasoning="on"`, calls 2 and 3 used no reasoning tokens. What does that do to the case for a sandwich on this model? Now compare the `cache read` column on call 2 between the two runs. Changing the thinking setting between turns invalidates Anthropic's cache for the conversation so far, so the sandwich paid to write it again.

## Part 2: Open Models (~25 min)

An open model is one whose weights you can download and run. Nothing is billed per token. You pay in hardware, electricity, seconds, and answer quality instead, and this part measures all four.

### 2.1 Install Ollama and create the two models

Install Ollama from https://ollama.com/download (macOS and Windows), or on Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Then, from `lesson4/lab/harness-lab`:

```bash
ollama --version                 # 0.40.4 or newer

ollama pull qwen3.5:4b           # the chat model: 3.3 GB (4.0 GB on Apple Silicon)
ollama create lab4-agent -f ollama/agent.Modelfile

ollama pull kev:4b-q4_K_M        # the decision model for Part 3: 2.7 GB
ollama create lab4-decide -f ollama/decide.Modelfile
```

Open `ollama/agent.Modelfile`. It has three instructions: which model, how large a context window, and a temperature of 0. The mesh serves whatever `lab4-agent` is as `mesh-local`, so the mesh needs no restart and no edit.

Try the model from the terminal, then see what it is using:

```bash
ollama run lab4-agent --think=false "In one sentence, what is a restocking fee?"
ollama ps
```

**Checkpoint 6:** `ollama ps` lists `lab4-agent` with `CONTEXT` 16384. The `PROCESSOR` column should say `100% GPU`. If it says CPU, the lab still works and every local step takes several times longer.

Now run the same command without `--think=false`. Qwen3.5 thinks before it answers unless told not to. On our machine the answer took 0.8 seconds without thinking and 69 seconds with it, for one sentence. The harness switches thinking off for this model on every call (`reasoning.off_value` in `config.yaml`) until you ask for it in 2.4.

**The context window is memory you pay for.** Ollama's default window depends on your hardware: 4,096 tokens on a machine with less than 24 GB of GPU memory. That is smaller than this lab's playbook plus tool schemas, so the Modelfile sets 16,384. On the build that Windows and Linux download, the model took 3.2 GB at 4,096 and 3.7 GB at 16,384.

**If memory is tight on a Mac**, change the `FROM` line to `qwen3.5:4b-q4_K_M` and run `ollama create` again. That is the build Windows and Linux use. On our Mac it stayed at 3.7 GB, where the default Apple Silicon build grew from 4.2 GB to 9.0 GB over the Part 4 experiment. It was a little slower (15.1 seconds per ticket against 12 to 14).

### 2.2 The same harness on your own machine

Nothing in the harness changes. Only the tier does. (Ollama unloads a model after five idle minutes, so the first call may also load it.)

```python
run = harness.run(TICKET, tier="local", stable_prefix=True, cap_tools=True, loop_guard=True)
```

On our run:

```
  This run                  $   0.0011   3 turns, 17.2s, ended by: done, unverified
    Input at full price     $   0.0000
    Cache writes            $   0.0000
    Cache reads             $   0.0000
    Output                  $   0.0000
    Your hardware           $   0.0011  100%  17.2s of your machine at $0.24/hour
```

**Checkpoint 7:** the model on your machine works the same tool loop, and every API line is \$0. Read the reply. On our run it got the rate limit right and both dollar amounts wrong (\$168.00 and \$60.00, where \$200.00 and \$180.00 are correct).

**Local is not free.** The mesh meters a local model at \$0 because there is no invoice. The ledger's "Your hardware" line charges for the seconds your machine was busy, at `local.usd_per_hour` in `config.yaml`. The default of \$0.24 is \$0.01 of electricity (60 W at \$0.17 per kWh) plus \$0.23 of hardware (a \$2,000 laptop over three years at eight hours a day). Use \$0.01 if the machine is already paid for and would otherwise sit idle. Use the sum if you are deciding whether to buy hardware to run models on.

**Think:** At \$0.24 an hour this attempt cost \$0.0011, against \$0.0080 for Haiku 4.5 on the same harness. That is seven times cheaper per attempt. A laptop runs one request at a time. What does that do to the comparison once you have 1,000 tickets an hour?

### 2.3 On your machine, a stable prefix saves seconds

Ollama keeps the prefix it has already processed in memory. There is no cache discount, because there is no bill. What you get back is time. Use a one-question ticket so the effect is easy to see, and run the second line twice:

```python
EASY = "Order NS-48213: I opened the espresso machine and it works fine, but I've decided to return it. How much will my refund be?"

run = harness.run(EASY, tier="local")                        # volatile line on top
run = harness.run(EASY, tier="local", stable_prefix=True)    # run this one twice
```

| On our run | Total | Call 1 | Call 2 |
| --- | ---: | ---: | ---: |
| Volatile line on top | 19.7 s | 8.2 s | 11.5 s |
| Stable prefix | 7.9 s | 1.5 s | 6.4 s |

**Checkpoint 8:** the `sec` column drops and the token columns do not change. Ollama does not report cached tokens, so the ledger shows every token "at full price" at a price of \$0. The saving is real and the only place to see it is the clock.

On a CPU the effect is larger. With the GPU switched off, the first call of a cold run took 37 seconds on our machine, most of it spent reading the playbook.

### 2.4 Let the open model think

In 2.2 the open model got two of three answers wrong. Turn its thinking back on:

```python
run = harness.run(TICKET, tier="local", stable_prefix=True, cap_tools=True, loop_guard=True, reasoning="on")
```

**Checkpoint 9:** on our run all three answers were right, in 22.9 seconds instead of 17.2 (\$0.0015 of hardware instead of \$0.0011). The first call wrote 505 output tokens instead of 178. The `reasoning` column still says 0, because Ollama does not report thinking tokens separately. They are in `output`.

**Think:** Thinking cost Haiku 4.5 70% more and changed nothing (1.6). It cost the open model 33% more time and fixed two answers. What does that suggest about where a reasoning budget is worth spending?

**Optional: break the context window.** Set `PARAMETER num_ctx 4096` in `ollama/agent.Modelfile`, run `ollama create lab4-agent -f ollama/agent.Modelfile` again, and repeat 2.2. With the `q4_K_M` build, the one Windows and Linux download, the request was refused on our machine with an HTTP 400 (`the prompt is longer than the context length`). The default Apple Silicon build did something different: it processed all 6,321 tokens and raised no error. Set the value back to 16384 and re-create the model when you are done.

## Part 3: Jev Decision Models (~25 min)

The closing pass in 1.6 fixed a wrong answer and made every ticket more expensive. Most drafts did not need it. This part puts a **gate** in front of the closing pass, so that only doubtful drafts pay for one, and compares two ways to build that gate: a chat model and a decision model. The decision model is Kev on your machine. Step 3.5 swaps in hosted Jev.

### 3.1 Your first decision

Ask the decision model from the terminal first. The state arrives on standard input and each `--ask` is one question:

```bash
echo "I was charged twice for order NS-48213 and I need one of the charges reversed today." | \
  ollama run lab4-decide \
    --ask "Is this message about a charge, an invoice, or a refund?" \
    --ask "Which team should handle this?" \
    --choice orders="shipping, delivery, and returns of physical items" \
    --choice billing="charges, invoices, refunds, and service credits" \
    --choice platform="API errors, rate limits, logins, and data retention"
```

On our run:

```
Is this message about a charge, an invoice, or a refund?  yes      78%
Which team should handle this?                            billing  82%
```

Each line is a decision and its probability. Nothing else is generated.

Now the same thing through the mesh, the way the harness will call it. Put this in a new file, `decide_test.py`:

```python
from harness import Harness

harness = Harness.from_config("config.yaml")

MESSAGE = (
    "I was charged twice for order NS-48213 and I need one of the charges reversed today. "
    "This is the second time I have had to write in about it."
)

decision = harness.mesh.decide(
    {"message": MESSAGE},
    {
        "is_billing": {"type": "noul", "instructions": "Is `message` about a charge, an invoice, or a refund?"},
        "queue": {
            "type": "choice",
            "instructions": "Which team should handle `message`?",
            "criteria": {
                "orders": "shipping, delivery, and returns of physical items",
                "billing": "charges, invoices, refunds, and service credits",
                "platform": "API errors, rate limits, logins, and data retention",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is `message`?",
            "criteria": [
                "a general question with no deadline",
                "wants an answer this week",
                "money or access is affected today",
                "threatens to cancel or to escalate",
            ],
        },
    },
)
for name, answer in decision.answers.items():
    print(f"{name:<11}{answer}")
print(f"{decision.input_tokens} input tokens, {decision.latency_s:.2f}s, ${decision.total_usd:.6f}, served by {decision.model}")
```

`mesh.decide` sends one `POST /v1/systemone` to the mesh with the model `mesh-decide`. The state is JSON here, and a question refers to one of its fields in backticks. There are three question types:

| Type | You supply | You get back (our run) |
| --- | --- | --- |
| `noul` | a yes/no question | the probability of yes: 0.851 |
| `choice` | labels, each with a description | the winning label, a probability for every label, and a confidence: `billing`, 0.908 of the probability, confidence 0.862 |
| `score` | ordered levels, lowest first | the expected level, a probability for every level, and a confidence: 1.53 on a scale of 0 to 3, confidence 0.451 |

**Checkpoint 10:** three typed answers from one request of 237 input tokens, in about half a second once the model is loaded (the first call also loads it). Run it again: the numbers are identical. There is no sampling and no temperature to set.

**Think:** The `urgency` answer puts 0.56 on "money or access is affected today" and 0.28 on "wants an answer this week". A chat model asked the same question would have written one word. What can you do with the distribution that you cannot do with the word?

### 3.2 A gate made from a chat model

The obvious gate is a judge: show a chat model the ticket, the evidence, and the draft, and ask whether the draft is right.

```python
run = harness.run(TICKET, stable_prefix=True, cap_tools=True, loop_guard=True, verify="llm")
```

**Checkpoint 11:** the trace has a call with the purpose `judge`, and the ledger has a new line, "Gate". On our run the judge read 2,883 input tokens at full price and cost \$0.0033, which was 20% of the run. Its prompt is new every time, so none of it is cached. The whole run cost \$0.0162, more than running the closing pass on everything (\$0.0128).

Read the `gate[llm]` note in the trace. On our run the judge did send the draft back, for the wrong reason: it objected that a service credit should not be "paid out as \$60.00 in cash". The closing pass corrected the amount anyway.

### 3.3 A gate made from a decision model

`verify="decision"` asks a decision model two yes/no questions about the draft: does it answer every question in the ticket, and does every figure in it follow from the evidence? The draft passes if the lower of the two probabilities is at least `decision.accept_at`. The questions are in `decisions.py`.

```python
run = harness.run(TICKET, stable_prefix=True, cap_tools=True, loop_guard=True, verify="decision")
```

**Checkpoint 12:** the trace has a `gate` line and a note such as `gate[decision]: P(complete)=0.89 P(supported)=0.65  passed`. On our run the gate read 4,400 input tokens. Once the model was loaded that took 4.4 seconds and \$0.0003 of hardware, about a tenth of what the judge cost (the first call also loads the model: 11.3 seconds). Then look at what it passed: the draft with the wrong \$60.00 credit went to the customer.

The gate was cheap, fast, and wrong. `accept_at: 0.5` was a guess.

**Break it on purpose.** Run the same line without `cap_tools=True`. The gate reads the evidence the agent saw, and uncapped that is about 16,800 tokens against the decision model's window of 8,192. A chat runtime may quietly drop the start of a prompt that is too long. A decision request is refused instead, because an answer about half the evidence would look exactly like a real one. The trace says so, and the harness runs the closing pass.

**Think:** The model said 0.65, not "yes". Where did the yes come from?

### 3.4 Choose the threshold from data

`fixtures/decisions.json` holds 66 first drafts this agent really wrote, each with the evidence it was written from and a label: did the reply pass the ticket's checks? 45 are good and 21 are bad. `calibrate.py` scores every one with the decision model and shows what each threshold would have let through.

```bash
python calibrate.py                    # about 3 minutes on a GPU; add --limit 20 on a slow machine
python calibrate.py --max-risk 0.10    # reuses the cached scores
```

On our run, the second command printed (some rows left out):

```
Decision model mesh-decide: 66 drafts (45 good, 21 bad)
  per draft: 2,254 input tokens, 2.3s, $0.00016

  accept_at   passes  coverage  bad passed   risk  bad caught
      0.500    62/66       94%          18    29%         14%
      0.600    58/66       88%          15    26%         29%
      0.650    50/66       76%          12    24%         43%
      0.700    33/66       50%           4    12%         81%
      0.722    28/66       42%           2     7%         90%   <- suggested
      0.750    20/66       30%           2    10%         90%
      0.800    13/66       20%           0     0%        100%
```

**Coverage** is the share of drafts that pass the gate: closing passes you do not pay for. **Risk** is the share of passed drafts that were bad: errors you ship. At 0.5 the gate passes nearly everything, including 18 of the 21 bad drafts. With no risk budget at all (`--max-risk 0`, the default) the script suggests 0.763, where only 27% of drafts skip the closing pass.

Now run the judge over the same drafts (about \$0.15):

```bash
python calibrate.py --max-risk 0.10 --judge
```

| On our run, 66 drafts | Passes | Bad drafts passed | Per draft | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Decision model at 0.722 | 28 (42%) | 2 | \$0.00016 | 2.3 |
| Chat-model judge | 28 (42%) | 4 | \$0.00234 | 1.8 |

The judge has no threshold to move. It passed the same number of drafts, let twice as many bad ones through, and cost about 15 times as much per draft.

Set the threshold the script suggested for **your** run in `config.yaml` (we used the `--max-risk 0.10` suggestion):

```yaml
decision:
  accept_at: 0.722
```

Run 3.3 again.

**Checkpoint 13:** the same draft now scores below the threshold (`P(supported)=0.65` on our run), the closing pass runs, and the reply says \$180.00. The run cost \$0.0126, of which the gate was \$0.0003.

**Think:** The 66 drafts were written about the same tickets you will run in Part 4. What does that do to your confidence in the threshold? "2 bad drafts out of 28 passed" is a small sample: how many labeled drafts would you want before promising a customer a 7% error rate? The threshold belongs to this decision model, these two questions, and this workload. Name three changes that would mean running `calibrate.py --rescore`.

### 3.5 Optional: hosted Jev

Kev runs on your machine. Jev is the hosted decision model from TypeSafe AI, and the swap is one alias:

1. Uncomment the `mesh-jev` block in `mesh_config.yaml`, export `TYPESAFE_API_KEY` in the mesh's terminal, and restart LiteLLM.
2. In `config.yaml`, set `decision.alias: mesh-jev` and `decision.local: false`.
3. Run `python calibrate.py --rescore`, and set `accept_at` from the new table. A threshold does not carry over from one model to another.

We did not have a TypeSafe key, so this step is untested by us. At Jev's list price of \$0.042 per million input tokens, the average draft above (2,254 tokens) would cost about \$0.0001.

## Part 4: The Cost Experiment (~20 min)

Parts 1-3 ran one ticket at a time. This part runs all twelve through eight harnesses and reports the number that matters: dollars per **solved** ticket.

### 4.1 Run the experiment

`run_experiment.py` ignores the switches you were toggling and builds eight clean arms. The dials (`tool_cap_tokens`, `accept_at`, the loop-guard budgets) still come from `config.yaml`:

| Arm | Model | Stable prefix | Capped tools | Loop guard | Closing pass |
| --- | --- | --- | --- | --- | --- |
| A naive | Haiku 4.5 | off | off | off | none |
| B + stable prefix | Haiku 4.5 | on | off | off | none |
| C + capped tools | Haiku 4.5 | on | on | off | none |
| D + loop guard | Haiku 4.5 | on | on | on | none |
| E + closing pass | Haiku 4.5 | on | on | on | every draft |
| F + judge gate | Haiku 4.5 | on | on | on | drafts the judge faults |
| G + decision gate | Haiku 4.5 | on | on | on | drafts below `accept_at` |
| H open model | qwen3.5:4b on your machine | on | on | on | every draft |

```bash
python run_experiment.py
```

This takes about 13 minutes and costs about \$1.00. To run some arms only, use `--arms A,D`. Per-ticket results go to `results.json`, and re-running some arms updates those arms and keeps the rest.

On our run (`accept_at: 0.722`):

```
  Arm                   Solved  Success  $/attempt    sec  $/solved    vs A
  A  naive               10/12      83%     0.0353    6.3    0.0423    100%
  B  + stable prefix     10/12      83%     0.0117    6.2    0.0140     33%
  C  + capped tools       8/12      67%     0.0071    5.8    0.0107     25%
  D  + loop guard         9/12      75%     0.0048    4.1    0.0065     15%
  E  + closing pass      11/12      92%     0.0072    7.4    0.0079     19%
  F  + judge gate        11/12      92%     0.0086    7.7    0.0094     22%
  G  + decision gate     12/12     100%     0.0065    8.6    0.0065     15%
  H  open model           8/12      67%     0.0008   12.3    0.0012      3%

Where the money went (share of each arm's spend):
  Arm                    Input at full price   Cache writes   Cache reads   Output   Gate   Your hardware
  A  naive                               93%             0%            0%       7%     0%              0%
  B  + stable prefix                      0%            55%           24%      20%     0%              0%
  C  + capped tools                       0%            33%           37%      30%     0%              0%
  D  + loop guard                         0%            42%           26%      31%     0%              0%
  E  + closing pass                       0%            32%           24%      43%     0%              0%
  F  + judge gate                         0%            25%           17%      33%    25%              0%
  G  + decision gate                      0%            34%           24%      40%     2%              0%
  H  open model                           0%             0%            0%       0%     0%            100%

Closing passes run:  A 0/12   B 0/12   C 0/12   D 0/12   E 11/12   F 5/12   G 6/12   H 12/12
```

`$/solved` is `$/attempt` divided by the success rate: a failed ticket has to be attempted again. Across our three runs arm G solved 12 of 12 every time, arm F solved 12, 12, and 11, arm E solved 11, 12, and 11, and the open model solved 6, 6, and 8.

If you have Langfuse from Lab 1, filter to the tag `lab4` and compare `arm-A` with `arm-G`. Each trace is also tagged with its ticket id.

### 4.2 Price a failed ticket

So far a failed ticket has cost nothing except another attempt. In a support queue a wrong reply gets reopened and a person does the work. Re-price the saved runs without calling a model:

```bash
python run_experiment.py --replay --fail-cost 0.25
python run_experiment.py --replay --fail-cost 3
```

On our run, dollars per solved ticket:

| Arm | Failure costs \$0 | Failure costs \$0.25 | Failure costs \$3 |
| --- | ---: | ---: | ---: |
| A naive | 0.0423 | 0.0923 | 0.6423 |
| B + stable prefix | 0.0140 | 0.0640 | 0.6140 |
| C + capped tools | 0.0107 | 0.1357 | 1.5107 |
| D + loop guard | 0.0065 | 0.0898 | 1.0065 |
| E + closing pass | 0.0079 | 0.0306 | 0.2806 |
| F + judge gate | 0.0094 | 0.0321 | 0.2821 |
| G + decision gate | 0.0065 | 0.0065 | 0.0065 |
| H open model | 0.0012 | 0.1262 | 1.5012 |

The cheapest arm at \$0 is the second most expensive at \$0.25. On our run the open model beat arm E only while a failed ticket cost less than two cents to clean up. Arm G's row does not move because it solved all twelve tickets on this run. Twelve tickets are not a guarantee.

### 4.3 Food for Thought Questions

**Question 1:** Arm C is cheaper per attempt than arm B and solved fewer tickets on every one of our runs. Open `results.json` and compare `failed_checks` for the same ticket in arms B and C. Which tickets did capping break? How would you find out whether the cap removed something the model needed, or whether the model simply stopped showing its working? (`--repeats 3` gives steadier success rates.)

**Question 2:** Arms E, F, and G run the same closing pass on 11, 5, and 6 drafts. The judge took 25% of arm F's spend and the decision model took 2% of arm G's. What does a gate have to be good at for gating to beat "check everything"? Arm E needs no labeled drafts and no second model. What is that simplicity worth?

**Question 3:** Arm G's threshold was chosen on drafts written about these same twelve tickets, so its 12 of 12 was measured partly on the data it was tuned on. Design the check you would run before trusting it in production. What would you monitor afterwards to notice that the threshold had gone stale?

**Question 4:** At \$0 per failure the open model costs 3% of the naive arm per solved ticket. At \$0.25 it costs more than the naive arm. Name a workload where a wrong answer really does cost less than two cents. For a workload where it does not, what would you try first: thinking on (2.4), a larger model, or the decision gate in front of an API model's closing pass?

**Question 5:** Based only on your tables, which harness would you ship for this workload, and which line of "Where the money went" would you attack next? What does this experiment not measure that could change your answer?

---

## Advanced Exercise: Build a Decision Model From a Chat Model's First Token

In Part 3 you sent typed questions to a decision model and got back probabilities. It looked like a different kind of model. It is mostly a different way of *reading* a model.

Before a language model produces a token, it computes a probability for every token it could produce, and a chat API shows you only the one that was picked. Ask for `logprobs`, stop after one token, and you have the outline of a decision: a probability for each permitted answer, in one pass, with nothing generated. In this exercise you build the three System One question types on top of that one token, repair the confidence with a temperature, and work out what a threshold on it is worth in dollars. No mesh, no network, no spend: everything runs on recordings of the Part 2 chat model.

### Learning objectives

By the end you will be able to explain, from first principles:

1. How a yes/no, a choice, and a score answer are each read from one token's probabilities
2. What the `confidence` in a System One answer is, as a formula you have implemented and checked against the real API
3. Why a confident model can still be badly calibrated, and how one fitted number repairs that without changing an answer
4. How a threshold on a cheap model's confidence trades coverage against errors shipped and dollars per decision

### The mental model

- **A decision is a read, not a generation.** One input pass and one token's worth of output. That is why it is fast and why it is priced on input only.
- **Only permitted answers count.** The model spreads probability over `yes`, `Yes`, ` yes`, and tokens that are not answers at all. A decision keeps the permitted answers and renormalizes.
- **A confidence is a formula.** For a choice it is how far the winner stands above an even split. For a score it is how tightly the probability sits around the most likely level. The tests check your formulas against 54 answers the real API returned.
- **A confidence is only useful if it is calibrated.** Temperature scaling is one number, fit on a few dozen labeled examples.
- **A threshold is a price.** Lower it and more decisions are taken on the cheap model's word, and more of them are wrong.
- **Training is the difference.** A purpose-built decision model keeps this shape and replaces the crude parts with trained ones. The demo shows how much that buys.

### Your task

Edit only `lesson4/lab/decision-model-lab/decision_model.py`. Implement the eight functions marked `TODO(student)`, in order:

| Function | Purpose |
| --- | --- |
| `option_probs` | Candidate tokens to a probability for each permitted answer |
| `temper` | Temperature scaling of a distribution |
| `noul` | A yes/no question to P(yes) |
| `choice` | A multiple-choice question to an answer, probabilities, and a confidence |
| `score` | An ordered scale to a level, an expected level, and a confidence |
| `expected_calibration_error` | Does "80% confident" mean right 80% of the time? |
| `fit_temperature` | The one number that repairs a model's confidence |
| `cascade` | What a threshold does to coverage, errors shipped, and cost |

Everything else is provided: the recordings, `auc`, `nll`, and the demo. Each function has a docstring spelling out the contract. More background is in [`decision-model-lab/README.md`](decision-model-lab/README.md).

### Workflow

```bash
cd lesson4/lab/decision-model-lab
source ../harness-lab/.venv/bin/activate    # the hands-on lab's venv
pip install -r requirements.txt             # adds pytest

# 1. Implement the TODOs in decision_model.py, checking your work as you go:
pytest -v

# 2. Once green, see what your decision model is worth next to the purpose-built one:
python run_demo.py
```

On our run with the reference solution:

| | One token of qwen3.5:4b | Kev 4B |
| --- | ---: | ---: |
| Yes/no on the 66 labeled drafts: AUC | 0.78 | 0.80 |
| Choice, 48 messages into 8 categories: correct | 41/48 | 47/48 |

The chat model was overconfident on the choice questions: 0.99 confident on average when right, and 0.70 when wrong. A temperature of 1.39, fit on half the messages, moved the calibration error on the other half from 0.113 to 0.079.

### Use your own model

Two optional steps talk to Ollama on your machine. Neither needs the mesh, because `logprobs` is on Ollama's OpenAI-compatible endpoint:

```bash
# One new message, classified live by the Part 2 chat model through your functions:
python run_demo.py --live "We were double charged and then locked out of the dashboard."

# Record your own model instead of ours (about two minutes), then rerun the demo:
python record_logprobs.py --model lab4-agent
```

### Definition of done

1. All tests in `test_decision_model.py` pass (`pytest -v`), including the ones that compare your `choice` and `score` confidence with what the real API returned.
2. `python run_demo.py` prints an AUC for your yes/no reader next to Kev's, a fitted temperature above 1, and a cascade table in which a higher threshold lowers both coverage and errors shipped.
3. You can explain why fitting a temperature changed the calibration error and did not change a single answer.

If you run out of gas or just want to run a completed solution, you can find the solution in `reference_solution.py` (`DECISION_MODEL_IMPL=reference_solution python run_demo.py`).

### References

- Exercise README: [`decision-model-lab/README.md`](decision-model-lab/README.md)
- On Calibration of Modern Neural Networks (temperature scaling, expected calibration error): https://arxiv.org/abs/1706.04599
- Ollama's OpenAI-compatible API (`logprobs`, `top_logprobs`): https://docs.ollama.com/api/openai-compatibility
- Kev 4B model card (how a purpose-built decision model is put together): https://huggingface.co/jaredpalmer/kev-4b

## References

- Lab 1 mesh (required): [`../../lesson1/lab/exercise_overview.md`](../../lesson1/lab/exercise_overview.md)
- Lab 2 index and compression (required): [`../../lesson2/lab/exercise_overview.md`](../../lesson2/lab/exercise_overview.md)
- Lecture demo, the harness ledger with a mock model: [`../lecture/harness_cost_savings.py`](../lecture/harness_cost_savings.py)
- The harness: [`harness-lab/harness.py`](harness-lab/harness.py). The gates: [`harness-lab/decisions.py`](harness-lab/decisions.py)
- LiteLLM prompt caching: https://docs.litellm.ai/docs/completion/prompt_caching
- LiteLLM decision models (`/v1/systemone`): https://docs.litellm.ai/docs/decisions
- LiteLLM with Ollama: https://docs.litellm.ai/docs/providers/ollama
- LiteLLM with TypeSafe (Jev): https://docs.litellm.ai/docs/pass_through/typesafe
- TypeSafe API: https://docs.typesafe.ai/api
- Ollama downloads: https://ollama.com/download
- Ollama requirements by platform: https://docs.ollama.com/macos, https://docs.ollama.com/windows, https://docs.ollama.com/linux, https://docs.ollama.com/gpu
- Ollama decision models and the System One API: https://docs.ollama.com/capabilities/decision, https://docs.ollama.com/api/systemone
- Ollama context length, thinking, and Modelfiles: https://docs.ollama.com/context-length, https://docs.ollama.com/capabilities/thinking, https://docs.ollama.com/modelfile
- The models: https://ollama.com/library/qwen3.5, https://ollama.com/library/kev
