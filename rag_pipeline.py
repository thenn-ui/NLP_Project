from kfp import dsl
from components.summarize_component import summarize_text_op

@dsl.pipeline(
    name="rag-summarization-pipeline",
    description="Summarize input text using a transformer"
)
def rag_pipeline(text: str):
    summarize_task = summarize_text_op(text)
