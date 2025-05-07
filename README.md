# NLP_Project
## Project Kickoff

This is a starter repository for our project.


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

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/breifly.git
cd breifly

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # or .\venv\Scripts\activate on Windows

### 3. Install Requirements

```bash
pip install -r requirements.txt


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
