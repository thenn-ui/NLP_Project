# components/summarize_component.py
from kfp.components import create_component_from_func

def summarize_text(text: str) -> str:
    from transformers import pipeline
    import textwrap

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = textwrap.wrap(text, 2000, break_long_words=False, break_on_hyphens=False)

    summaries = []
    for chunk in chunks:
        result = summarizer(chunk, max_length=512, min_length=100, do_sample=False)
        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)

summarize_text_op = create_component_from_func(
    summarize_text,
    base_image="python:3.9",
    packages_to_install=["transformers", "torch", "textwrap3"]
)
