from kfp import compiler
from rag_pipeline import rag_pipeline

compiler.Compiler().compile(
    pipeline_func=rag_pipeline,
    package_path="rag_pipeline.yaml"
)
