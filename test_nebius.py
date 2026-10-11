"""First test of Nebius Token Factory: list the NVIDIA Nemotron models, then call one.

Setup:
    pip install openai
    export NEBIUS_API_KEY="paste-your-key-here"      # Mac/Linux
    setx NEBIUS_API_KEY "paste-your-key-here"        # Windows (then reopen the terminal)

Run:
    python test_nebius.py
    MODEL="exact/model-id" python test_nebius.py     # to pick a specific model
"""
import os
import sys

from openai import OpenAI

api_key = os.environ.get("NEBIUS_API_KEY")
if not api_key:
    sys.exit("NEBIUS_API_KEY is not set. See the setup notes at the top of this file.")

client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=api_key,
)

# 1. Ask the API which models your account can use, and show the Nemotron ones.
all_ids = [m.id for m in client.models.list().data]
nemotron_ids = [m for m in all_ids if "nemotron" in m.lower()]

print("Nemotron models available to you:")
for model_id in nemotron_ids:
    print("  ", model_id)

if not nemotron_ids and not os.environ.get("MODEL"):
    sys.exit("No Nemotron models found. Check the playground for exact model names.")

# 2. Send one test message to the chosen model.
model = os.environ.get("MODEL") or nemotron_ids[0]
print(f"\nCalling: {model}\n")

response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "In one sentence, what is an AI agent?"}],
    temperature=0.6,
    max_tokens=2000,
)

choice = response.choices[0]
message = choice.message
print("finish_reason:", choice.finish_reason)
print("answer:", message.content)

# Some Nemotron models "think" first. That thinking can use up the token limit
# before an answer appears, so show it if the API returns it.
reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
if reasoning:
    print("\nreasoning (first 300 characters):", reasoning[:300])