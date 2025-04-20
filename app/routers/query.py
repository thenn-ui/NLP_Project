from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.utils.document_utils import extract_text_from_file
from app.services.rag_qa import answer_question

router = APIRouter(prefix="/query", tags=["Question Answering"])

@router.post("")
async def query_doc(
    file: UploadFile = File(None),
    text: str = Form(None),
    question: str = Form(...),
    top_k: int = Form(3)
):
    # Validate input
    if file:
        try:
            content = await file.read()
            doc_text = extract_text_from_file(file.filename, content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"File read error: {str(e)}")
    elif text:
        doc_text = text
    else:
        raise HTTPException(status_code=400, detail="Provide either a file or raw text.")

    # Get answer from RAG pipeline
    try:
        response = answer_question(doc_text, question, k=top_k)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while answering: {str(e)}")
