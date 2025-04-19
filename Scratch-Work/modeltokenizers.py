from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


classifier = pipeline("sentiment-analysis")

res = classifier("Drake and Josh! Well that's the golden Nickelodeon Era...")

print(res)

###########################

# behind the scenes for the pipeline


model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("Drake and Josh! Well that's the golden Nickelodeon Era...")
print(res)

####################################################

sequence = "Getting started with HuggingFace in 15 minutes. Transformers, Pipeline, Tokenizer, Models"

res = tokenizer(sequence)
print(res)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)