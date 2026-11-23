from harness import Harness, print_ledger, print_trace

# One ticket, three unrelated questions. Answering it takes four records
# and three policy lookups, which is enough turns for a harness to matter.
TICKET = (
    "Hi, this is Dana at Acme Outfitters (account ACME-77). Three things. First, on order NS-48213 "
    "I opened the espresso machine and it works, but we want to return it: how much will the refund "
    "be? Second, your API was unreliable in October 2026: what service credit do we get, in dollars? "
    "Third, we keep getting 429 errors on our production key: what is our requests-per-minute limit, "
    "and how far over it were we at peak?"
)

harness = Harness.from_config("config.yaml")

# 1. Run TICKET through the harness as configured (every switch off).
# 2. Print the ledger and the trace for that run.
# 3. Print the reply the customer would get.
#
# Hint: harness.run(TICKET) returns a Run.
#       print_ledger(harness, run) shows what the run cost, by line item.
#       print_trace(run) shows every model call and every tool call.
#       Useful fields: .answer, .turns, .cost_usd, .latency_s, .stop_reason,
#       .prompt_tokens, .cached_tokens, .cache_write_tokens, .completion_tokens,
#       .calls, .tool_log, .events
#       Any switch in config.yaml's `harness:` section can be overridden for
#       one run: harness.run(TICKET, stable_prefix=True)
