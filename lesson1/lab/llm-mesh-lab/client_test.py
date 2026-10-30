from openai import OpenAI

QUESTION = "What is the capital of France?"

# Note: this is the STANDARD OpenAI Protocol SDK
client = OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-mesh-lab-1234",
)

# Loop through the model aliases and print the answer to QUESTION for each.
