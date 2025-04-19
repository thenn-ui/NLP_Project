from transformers import pipeline


classifier = pipeline("sentiment-analysis")

res = classifier("Drake and Josh! Well that's the golden Nickelodeon Era...")

print(res)