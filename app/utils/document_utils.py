import fitz  # PyMuPDF
import os
from io import BytesIO

def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def extract_text_from_pdf(file_bytes: bytes) -> str:
    pdf_stream = BytesIO(file_bytes)
    doc = fitz.open(stream=pdf_stream, filetype="pdf")

    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"

    return full_text.strip()
