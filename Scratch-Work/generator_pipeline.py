from transformers import pipeline


generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "Drake and Josh! Well that's the",
    max_length=30,
    num_return_sequences=2,
    )

print(res)