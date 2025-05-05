import streamlit as st
import pdfplumber
import io
import numpy as np
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import requests

# --- Streamlit Setup ---
st.set_page_config(page_title="📚 iAssist Chat & Summary", layout="wide")
st.title("📚 iAssist: Summarize & Chat with Documents")

# --- MongoDB Setup ---
mongo_uri = "mongodb+srv://devarshpathak7:xzMSjFPsu06CNSuS@cluster0.rkzlstf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
db = client["summary_db"]
summary_col = db["summaries"]
chunk_col = db["doc_chunks"]

# --- Groq Setup ---
api_key = "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"
groq_client = Groq(api_key=api_key)

# --- Hugging Face API ---
HF_API_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
headers = {"Authorization": "Bearer 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"}

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


def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
        return text.strip()
    elif uploaded_file.type == "text/plain":
        return uploaded_file.getvalue().decode("utf-8")
    return None

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

def split_text(text, max_chunk_words=700):
    words = text.split()
    return [' '.join(words[i:i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]

def summarize_large_text(text):
    chunks = split_text(text)
    summaries = []
    for i, chunk in enumerate(chunks):
        st.info(f"Summarizing chunk {i + 1} of {len(chunks)}...")
        summary = summarize_text(chunk)
        summaries.append(summary)
    return "\n\n".join(summaries)

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

# --- Sidebar Controls ---
st.sidebar.header("Options")
action = st.sidebar.radio("Choose an action", ["Summarize", "Chat with Document"])

file_list = [doc["filename"] for doc in summary_col.find({}, {"filename": 1, "_id": 0})]
selected_file = st.sidebar.selectbox("Previously Uploaded Files", options=["-- Select --"] + file_list)

# --- Show Previous Summary ---
if selected_file != "-- Select --":
    doc = summary_col.find_one({"filename": selected_file})
    if doc:
        st.markdown("### 📄 Retrieved Summary from Database")
        st.subheader(f"Summary for: `{selected_file}`")
        st.write(doc["summary"])
        st.markdown("---")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt"])

if uploaded_file:
    filename = uploaded_file.name
    st.markdown(f"### 📁 File Uploaded: `{filename}`")

    with st.spinner("Extracting text..."):
        full_text = extract_text_from_file(uploaded_file)

    if not full_text:
        st.error("❌ No extractable text found.")
    else:
        st.success("✅ Text extracted.")

        if action == "Summarize":
            existing = summary_col.find_one({"filename": filename})
            if existing:
                st.success("✅ Summary already exists in DB")
                st.subheader("📄 Summary:")
                st.write(existing["summary"])
            else:
                if st.button("🔄 Generate Summary"):
                    with st.spinner("⏳ Summarizing..."):
                        summary = summarize_large_text(full_text)

                    summary_col.insert_one({
                        "filename": filename,
                        "summary": summary
                    })

                    st.success("✅ Summary generated and saved to database!")
                    st.subheader("📄 Final Summary:")
                    st.write(summary)

        elif action == "Chat with Document":
            chunks = chunk_text(full_text, max_words=200)
            store_chunks(chunks, filename)
            st.success("✅ Document chunked and stored.")

            user_query = st.text_input("Ask something about this document:")
            if user_query:
                relevant_chunks = retrieve_relevant_chunks(user_query, filename)
                if relevant_chunks:
                    context = "\n\n".join(relevant_chunks)
                    response = ask_groq(context, user_query)
                    st.markdown("### 💬 Response")
                    st.write(response)
                else:
                    st.warning("No relevant chunks found.")
