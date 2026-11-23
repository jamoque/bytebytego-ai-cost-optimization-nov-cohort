#!/usr/bin/env python3
"""
How fixtures/recorded.json was made (PROVIDED). Run it to record your own model.

    python record_logprobs.py                        # the lab's local chat model, on Ollama's default port
    python record_logprobs.py --model qwen3.5:9b     # any chat model Ollama serves
    python record_logprobs.py --limit 10             # a quick look

A decision model answers in one forward pass. So can any chat model, if you
stop it after one token and look at the probabilities it assigned to every
token it could have said instead. This script does exactly that, 114 times:

    66 yes/no questions    "is every figure in this reply supported by the evidence?"
                           about the labeled drafts in ../harness-lab/fixtures/decisions.json
    48 choice questions    "which of these eight categories is this message?"
                           about the labeled messages in fixtures/messages.json

For each one it asks for ONE output token with `logprobs` on and keeps the
20 most likely first tokens with their log-probabilities. Nothing else is
stored: no generated text, because none is generated.

It talks to Ollama's OpenAI-compatible endpoint directly. The mesh is not
involved, because the chat-completions route is where `logprobs` lives.
"""

from __future__ import annotations

import argparse
import json
import string
import time
from pathlib import Path

import httpx

HERE = Path(__file__).parent
DRAFTS = HERE.parent / "harness-lab" / "fixtures" / "decisions.json"
MESSAGES = HERE / "fixtures" / "messages.json"
OUT = HERE / "fixtures" / "recorded.json"

NOUL_SYSTEM = "You answer one question about the material you are given. Reply with exactly one word: yes or no."
NOUL_QUESTION = (
    "QUESTION: Is every amount, date, and number stated in REPLY supported by EVIDENCE?\n"
    "yes: each figure in the reply appears in the evidence or follows directly from it.\n"
    "no: the reply states a figure that the evidence does not support or contradicts."
)
CHOICE_SYSTEM = "You sort a customer message into exactly one category. Reply with exactly one letter."


def noul_prompt(draft: dict) -> str:
    evidence = "\n\n".join(draft["evidence"])
    return f"TICKET:\n{draft['ticket']}\n\nEVIDENCE:\n{evidence}\n\nREPLY:\n{draft['reply']}\n\n{NOUL_QUESTION}"


def choice_prompt(text: str, options: dict[str, str]) -> tuple[str, dict[str, str]]:
    letters = dict(zip(string.ascii_uppercase, options))
    menu = "\n".join(f"{letter}. {name}: {options[name]}" for letter, name in letters.items())
    return f"MESSAGE:\n{text}\n\nCATEGORIES:\n{menu}\n\nWhich category does MESSAGE belong to?", letters


def first_token(client: httpx.Client, model: str, system: str, user: str, top: int) -> tuple[list[dict], float]:
    """Ask for one token and return the `top` most likely candidates for it."""
    t0 = time.perf_counter()
    response = client.post(
        "/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": top,
            "reasoning_effort": "none",      # a thinking model's first token would be <think>, not an answer
        },
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["logprobs"]["content"]
    candidates = [{"token": c["token"], "logprob": round(c["logprob"], 4)} for c in content[0]["top_logprobs"]]
    return candidates, time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="lab4-agent")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--top", type=int, default=20, help="candidates to keep per question (Ollama allows up to 20)")
    parser.add_argument("--limit", type=int, default=0, help="record only the first N of each kind")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    drafts = json.loads(DRAFTS.read_text())["drafts"]
    triage = json.loads(MESSAGES.read_text())
    options: dict[str, str] = triage["options"]
    messages = triage["messages"]
    if args.limit:
        drafts, messages = drafts[: args.limit], messages[: args.limit]

    noul, choice, seconds = [], [], 0.0
    with httpx.Client(base_url=args.base_url, timeout=600, headers={"Authorization": "Bearer ollama"}) as client:
        for i, draft in enumerate(drafts, start=1):
            top, dt = first_token(client, args.model, NOUL_SYSTEM, noul_prompt(draft), args.top)
            seconds += dt
            noul.append({"id": draft["id"], "label": draft["good"], "top": top})
            if i % 10 == 0 or i == len(drafts):
                print(f"  yes/no {i}/{len(drafts)}", flush=True)
        letters: dict[str, str] = {}
        for i, message in enumerate(messages, start=1):
            prompt, letters = choice_prompt(message["text"], options)
            top, dt = first_token(client, args.model, CHOICE_SYSTEM, prompt, args.top)
            seconds += dt
            choice.append({"id": message["id"], "text": message["text"], "label": message["label"], "top": top})
            if i % 10 == 0 or i == len(messages):
                print(f"  choice {i}/{len(messages)}", flush=True)

    record = {
        "description": (
            "First-token candidates from a chat model, one forward pass per question. `top` holds the most likely "
            "first tokens and their natural-log probabilities. `noul` items are yes/no questions about the labeled "
            "drafts in ../harness-lab/fixtures/decisions.json (label true = the draft was good). `choice` items ask "
            "which category a message belongs to; the model was shown the categories as lettered options, and "
            "`letters` maps each letter back to its category."
        ),
        "model": args.model,
        "top_logprobs": args.top,
        "seconds_per_question": round(seconds / max(len(noul) + len(choice), 1), 2),
        "letters": letters,
        "options": options,
        "noul": noul,
        "choice": choice,
    }
    Path(args.out).write_text(json.dumps(record, indent=1, ensure_ascii=False))
    print(f"Wrote {args.out}: {len(noul)} yes/no and {len(choice)} choice questions, {record['seconds_per_question']}s each")


if __name__ == "__main__":
    main()
