from pipeline import Pipeline, token_count

QUESTION = "How do I get a refund for a damaged order?"

pipeline = Pipeline.from_config("config.yaml")

# 1. Retrieve chunks for QUESTION.
# 2. Print each hit's source, score, token count, and a short text preview.
# 3. Print the stuffed-corpus token count vs. the retrieved-chunk token count.
# 4. Call pipeline.ask(QUESTION) and print the mesh answer, cache_hit,
#    semantic_similarity, llm_calls, and prompt_tokens. This always goes
#    through the Lab 1 mesh; the gateway decides hit vs miss.
#
# Hint: pipeline.retrieve(QUESTION) returns a list of Hit objects.
#       pipeline.corpus_token_count() is the naive-stuffing baseline.
#       token_count(hit.text) (or hit.tokens) is the size of one chunk.
