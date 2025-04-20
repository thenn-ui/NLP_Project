from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from app.services.retriever import create_vector_store_from_text
import torch

# Build the LLM pipeline
def get_qa_pipeline():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        max_length=256,
        do_sample=False
    )

    return HuggingFacePipeline(pipeline=pipe)

# CoT prompt
cot_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question:
{question}

Let's think step by step.
"""
)

# Main function
def answer_question(doc_text: str, query: str, k: int = 3, use_cot: bool = False) -> dict:
    vectorstore: FAISS = create_vector_store_from_text(doc_text)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = get_qa_pipeline()

    if use_cot:
        print("[INFO] Using Chain-of-Thought reasoning.")
        cot_chain = LLMChain(llm=llm, prompt=cot_prompt)

        combine_chain = StuffDocumentsChain(
            llm_chain=cot_chain,
            document_variable_name="context"
        )


        rag_chain = RetrievalQA(
            retriever=retriever,
            combine_documents_chain=combine_chain,
            return_source_documents=True
        )
    else:
        print("[INFO] Using direct QA.")
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
