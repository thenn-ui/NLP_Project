import streamlit as st
import requests
import PyPDF2
import io

# Hugging Face API setup
API_URL = "https://api-inference.huggingface.co/models/Falconsai/text_summarization"
headers = {
    "Authorization": f"Bearer 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"
}

# Function to summarize text
def summarize_text(text):
    print(text)
    payload = {
        "inputs": text,
        "parameters": {
            "max_length": 50000,  # you can adjust based on your needs
            "min_length": 50,
            "do_sample": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]['summary_text']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Function to extract text from uploaded file
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    elif uploaded_file.type == "text/plain":
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        return stringio.read()
    else:
        return None

# Streamlit App
st.title("📚 Scientific Journal Summarizer")
st.write("Upload a **PDF** or **TXT** file and get a concise summary using Falconsai model.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt"])

if uploaded_file is not None:
    with st.spinner('Extracting text...'):
        text = extract_text_from_file(uploaded_file)

    if text:
        st.success('Text extracted successfully!')

        if st.button("Generate Summary"):
            with st.spinner('Summarizing...'):
                summary = summarize_text(text)
            st.subheader("Summary:")
            st.write(summary)
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")

