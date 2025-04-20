# app/services/rag_qa.py

from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from app.services.retriever import create_vector_store_from_text

import torch

# Load QA generation model
def get_qa_pipeline():
    model_name = "google/flan-t5-base"  # You can try larger or custom models too
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model = model,
        tokenizer = tokenizer,
        device = 0 if torch.cuda.is_available() else -1,
        max_length = 256,
        do_sample = False
    )

    return HuggingFacePipeline(pipeline=pipe) # converts hugging face pipeline into LangChain pipeline

# Main function to answer questions using RAG
def answer_question(doc_text: str, query: str, k: int = 3) -> str:
    vectorstore: FAISS = create_vector_store_from_text(doc_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = get_qa_pipeline() # flan-t5 model - works well for question answering tasks

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = rag_chain({"query": query})
    return {
        "answer": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]]
    }
