from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.summarizer import get_summary
from app.utils.document_utils import extract_text_from_file

router = APIRouter(prefix="/summarize", tags=["Summarization"])

@router.post("")
async def summarize(
    file: UploadFile = File(None),
    text: str = Form(None),
    max_length: int = Form(512),
    min_length: int = Form(100)
):
    # Handle input
    if file:
        try:
            content = await file.read()
            text_content = extract_text_from_file(file.filename, content)
            print("LOG: ", text_content)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")
    elif text:
        text_content = text
    else:
        raise HTTPException(status_code=400, detail="Please provide either a file or text.")

    # Summarize
    summary = get_summary(text_content, scale_factor=0.40)
    return {
        "summary": summary
    }
