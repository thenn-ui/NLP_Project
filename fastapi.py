from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from typing import List
import pdfplumber
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import uvicorn

app = FastAPI()

# --- MongoDB Setup ---
mongo_uri = "mongodb+srv://devarshpathak7:xzMSjFPsu06CNSuS@cluster0.rkzlstf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
db = client["summary_db"]
summary_col = db["summaries"]
chunk_col = db["doc_chunks"]

# --- Groq Setup ---
api_key = "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"
groq_client = Groq(api_key=api_key)

# --- Helper Functions ---
def summarize_text(text):
    try:
        prompt = f"Summarize the following scientific or technical content clearly and concisely:\n\n{text}\n\nSummary:"
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error summarizing with Groq: {e}"

def extract_text_from_pdf(file: bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file)) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text.strip()

def chunk_text(text, max_words=200):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def store_chunks(chunks, filename):
    chunk_col.delete_many({"filename": filename})
    for i, chunk in enumerate(chunks):
        chunk_col.insert_one({"filename": filename, "chunk_index": i, "content": chunk})

def retrieve_relevant_chunks(user_query, filename):
    docs = list(chunk_col.find({"filename": filename}).sort("chunk_index", 1))
    texts = [doc["content"] for doc in docs]

    if not texts:
        return []

    vectorizer = TfidfVectorizer().fit(texts + [user_query])
    doc_vecs = vectorizer.transform(texts)
    query_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vec, doc_vecs).flatten()
    top_index = similarities.argmax()

    context_chunks = []
    if top_index > 0:
        context_chunks.append(texts[top_index - 1])
    context_chunks.append(texts[top_index])
    if top_index < len(texts) - 1:
        context_chunks.append(texts[top_index + 1])

    return context_chunks

def ask_groq(context, question):
    try:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error contacting Groq: {e}"

# --- API Endpoints ---
@app.post("/upload_and_summarize/")
async def upload_and_summarize(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()

    if filename.endswith(".pdf"):
        full_text = extract_text_from_pdf(contents)
    elif filename.endswith(".txt"):
        full_text = contents.decode("utf-8")
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

    existing = summary_col.find_one({"filename": filename})
    if existing:
        return {"filename": filename, "summary": existing["summary"]}

    chunks = chunk_text(full_text, max_words=700)
    summaries = [summarize_text(chunk) for chunk in chunks]
    final_summary = "\n\n".join(summaries)

    summary_col.insert_one({"filename": filename, "summary": final_summary})
    return {"filename": filename, "summary": final_summary}

@app.post("/upload_and_chat/")
async def upload_and_chat(file: UploadFile = File(...), question: str = Form(...)):
    filename = file.filename
    contents = await file.read()

    if filename.endswith(".pdf"):
        full_text = extract_text_from_pdf(contents)
    elif filename.endswith(".txt"):
        full_text = contents.decode("utf-8")
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported file type."})

    chunks = chunk_text(full_text, max_words=200)
    store_chunks(chunks, filename)
    relevant_chunks = retrieve_relevant_chunks(question, filename)
    if not relevant_chunks:
        return {"message": "No relevant information found."}

    context = "\n\n".join(relevant_chunks)
    response = ask_groq(context, question)
    return {"response": response}

# --- Run the server ---
# To run: `uvicorn filename:app --reload`
if __name__ == "__main__":
    uvicorn.run("filename:app", host="0.0.0.0", port=8000, reload=True)