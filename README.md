# NLP_Project

# 📚 BREIFLY: Scientific Document Summarizer & Chat Assistant

**BREIFLY** is a powerful web application built with Streamlit that allows users to upload, summarize, and interactively chat with scientific or technical documents. It leverages large language models from **Groq (LLaMA 3)** and integrates **MongoDB** for storage and **TF-IDF-based chunk retrieval** to power document-aware question answering.

---

## 🚀 Features

- 📄 Upload `.pdf` and `.txt` documents
- ✂️ Chunk large documents for better processing
- 🧠 Summarize content using Groq’s LLaMA 3
- 💬 Ask questions based on document context
- 🧾 MongoDB-backed chunk storage and retrieval
- 🎯 TF-IDF similarity search to find relevant context
- 🧪 Specifically optimized for **scientific or technical papers**

---

## 📦 Tech Stack

| Layer           | Tool/Library                        |
|----------------|-------------------------------------|
| Frontend       | Streamlit                           |
| LLMs           | Groq (LLaMA 3-70B)                  |
| Summarization  | Hugging Face (Fallback)             |
| Backend        | Python, pdfplumber, scikit-learn    |
| Database       | MongoDB Atlas                       |
| Vectorization  | TF-IDF (sklearn)                    |

---

## 🔑 API Keys and Database Configuration

### 🗃️ MongoDB Atlas
- Create a free MongoDB cluster at [https://www.mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas).
- Replace the `mongo_uri` in the script with your connection string.
- Ensure your IP address is whitelisted and database access is enabled.

### ⚡ Groq API
- Sign up at [https://console.groq.com](https://console.groq.com) to get your API key.
- Replace the `api_key` variable in your code with your Groq token.

### 🤖 Hugging Face API (Optional)
- Generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
- Replace the token in the `headers` dictionary in your script.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```console
git clone [https://github.com/thenn-ui/NLP_Project](https://github.com/thenn-ui/NLP_Project)
```

### 2. (Optional) Create a Virtual Environment

```console
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows
```
### 3. Install Requirements

```console
pip install -r requirements.txt
```

### In the App:
- Upload a `.pdf` or `.txt` file.
- Choose from:
  - 🔍 **Summarize** — Breaks content into manageable chunks and summarizes using LLM.
  - 💬 **Chat with Document** — Performs TF-IDF-based similarity search and queries Groq's LLaMA 3 with context.
- Stored summaries and document chunks can be reused for faster access.

---

## 🔒 Security Considerations
- API keys should **not** be hardcoded in production environments.
- Use environment variables or a `.env` file to store sensitive credentials.
- Ensure MongoDB access is IP-restricted and protected with authentication.

---

## 📌 Future Enhancements
- ✅ Integrate full-text vector search (e.g., FAISS or MongoDB Atlas Search)
- 🔁 Add user authentication and multi-user session support
- 📊 Build a dashboard for document usage and analytics
- 📤 Allow exporting summaries and chat sessions to PDF or Markdown
  
# 💡 Contribution Guidelines
**Feature Suggestions** : If you have ideas for new features or improvements, please create an issue on the GitHub repository outlining the feature request in detail. This will allow for discussion and collaboration among contributors.

**Code Contributions** : If you'd like to contribute code to the project, please follow these steps:

* Fork the repository to your GitHub account.
* Create a new branch for your feature or bug fix.
* Make your changes and ensure that they adhere to the project's coding style and guidelines.
* Write unit tests to cover your changes, if applicable.
* Submit a pull request to the main repository, explaining the purpose of your changes and any relevant details.
* Make sure to create docker images of your improvements and according to the format of current app.

**Documentation** : Clear and comprehensive documentation is essential for the project's users and contributors. You can contribute by:

* Improving existing documentation.
* Adding documentation for new features or components.
* Providing usage examples and tutorials.

**Performance Optimization** : If you have expertise in optimizing code or improving system performance, you can contribute by:

* Identifying and addressing performance bottlenecks.
* Refactoring code for better efficiency.
* Implementing caching mechanisms where applicable.

**Community Engagement** : Engage with other contributors and users by participating in discussions, providing feedback, and helping resolve issues users report.

Remember to adhere to the project's licensing terms and code of conduct while contributing. Your contributions are valuable and will help improve this project worldwide. Thank you for your interest and support!
