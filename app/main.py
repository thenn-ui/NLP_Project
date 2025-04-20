from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.services.summarizer import init_summarizer
from app.routers import summarize, query

app = FastAPI(title="SmartDocAI", version="0.1")


init_summarizer()

app.include_router(summarize.router)
app.include_router(query.router)


app.mount("/static", StaticFiles(directory="static"), name="static")

# routes

@app.get("/")
def root():
    return {"message": "SmartDocAI is running 🚀"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.ico")