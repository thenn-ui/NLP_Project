# app/services/retriever.py

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from typing import List

# Initialize embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def split_text_to_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_vector_store_from_text(text: str) -> FAISS:
    chunks = split_text_to_chunks(text)
    vectorstore = FAISS.from_texts(chunks, embedding=embedding_model)
    return vectorstore

def retrieve_top_k_docs(vectorstore: FAISS, query: str, k: int = 3) -> List[Document]:
    return vectorstore.similarity_search(query, k=k)
