import streamlit as st
import requests
import pdfplumber
import io
from pymongo import MongoClient

# -------------------- MongoDB Setup --------------------
mongo_uri = "mongodb+srv://devarshpathak7:xzMSjFPsu06CNSuS@cluster0.rkzlstf.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri, tlsAllowInvalidCertificates=True)
db = client["summary_db"]
collection = db["summaries"]

# -------------------- Hugging Face API Setup --------------------
API_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
headers = {
    "Authorization": f"Bearer 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"
}

# -------------------- Summarization Functions --------------------
def summarize_text(text):
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 5000,
            "min_length": 50,
            "do_sample": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['summary_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

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

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text
    elif uploaded_file.type == "text/plain":
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    else:
        return None

# -------------------- Streamlit App --------------------
st.set_page_config(page_title="📚 Summarizer with MongoDB", layout="wide")
st.title("📚 Scientific Journal Summarizer with MongoDB")
st.write("Upload a **PDF** or **TXT** file to generate a summary or chat with the document.")

# --------- Sidebar: Summary/Chat Toggle and File Selection ---------
st.sidebar.header("📁 Options & History")
action = st.sidebar.radio("Choose an action:", ["Summarize", "Chat with Document"])

all_files = collection.find({}, {"filename": 1, "_id": 0})
file_list = [doc["filename"] for doc in all_files]
selected_file = st.sidebar.selectbox("Previously Uploaded Files", options=["-- Select --"] + file_list)

# Show selected file's summary in the main area
if selected_file != "-- Select --":
    doc = collection.find_one({"filename": selected_file})
    if doc:
        st.markdown("### 📄 Retrieved Summary from Database")
        st.subheader(f"Summary for: `{selected_file}`")
        st.write(doc["summary"])
        st.markdown("---")

# --------- Main Upload Area ---------
st.header("📤 Upload a Document")
uploaded_file = st.file_uploader("Upload a new file", type=["pdf", "txt"])

if uploaded_file is not None:
    filename = uploaded_file.name
    st.markdown(f"### 📁 File Uploaded: `{filename}`")

    # Check DB for summary
    existing = collection.find_one({"filename": filename})

    with st.spinner('🔍 Extracting text...'):
        text = extract_text_from_file(uploaded_file)

    if not text:
        st.error("❌ Unsupported file type or no extractable text found.")
    else:
        st.success('✅ Text extracted successfully!')

        if action == "Summarize":
            if existing:
                st.success("✅ Summary already exists in database!")
                st.subheader("📄 Retrieved Summary:")
                st.write(existing["summary"])
            else:
                if st.button("🔄 Generate Summary"):
                    with st.spinner('⏳ Summarizing...'):
                        summary = summarize_large_text(text)

                    collection.insert_one({
                        "filename": filename,
                        "summary": summary
                    })

                    st.success("✅ Summary generated and saved to database!")
                    st.subheader("📄 Final Summary:")
                    st.write(summary)

        elif action == "Chat with Document":
            st.info("🧠 Chat functionality will be implemented here.")
            # Placeholder: Add your vector search + context + Groq API call here
