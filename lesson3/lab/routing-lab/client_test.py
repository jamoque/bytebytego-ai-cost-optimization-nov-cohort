from routing import Router, final_line

# Two questions about the same policy area. One is a lookup. One needs a
# multi-step derivation from the same documents.
EASY = "How many days after delivery do I have to request a refund?"
HARD = (
    "I mistyped my password 3 times at 10:00 a.m. and 2 more times at 10:20 a.m. "
    "Is my account locked?"
)

router = Router.from_config("config.yaml")

# 1. Ask EASY and HARD on the weak tier, then on the strong tier
#    (four calls in total).
# 2. For each result print: the strategy, the model that answered, prompt
#    and completion tokens, cost in USD, and the final answer line.
# 3. Print how many times more the strong tier cost than the weak tier for
#    the same question.
#
# Hint: router.ask(question, strategy="weak") returns a RouteResult.
#       Useful fields: .model, .tier, .prompt_tokens, .completion_tokens,
#       .cost_usd, .answer, .reason, .calls
#       final_line(result.answer) is the text after "Answer:".
#       strategy can be a tier (weak | mid | strong) or a router
#       (gateway | deterministic | cascade | preference).
