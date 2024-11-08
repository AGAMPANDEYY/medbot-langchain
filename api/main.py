# main.py
import logging
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from api.rag_pipeline import RAGPipeline

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)

load_dotenv(dotenv_path="./api/.env")

app=FastAPI()
pipeline= RAGPipeline()

pinecone_api_key = os.getenv("PINECONE_API_KEY")
if not pinecone_api_key:
        raise ValueError("Pinecone API key is missing. Please set it in the environment.")

# Define the path to your PDF and the question

pdf_path = r"C:\Agam\Work\medbot\dataset\Gynaecology-DC-Dutta.pdf"


# Define the path for the static directory using an absolute path
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    file_path = static_dir / "index.html"
    print(f"Serving index file from: {file_path}")  # Debugging line
    with open(file_path) as file:
        return HTMLResponse(file.read())
    
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    question = request.question
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    # Run inference through the medbot pipeline
    try:
        answer = str(pipeline.query_medbot(question))
        return {"question": question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/health")
async def healt_check():
     return {"status": "OK"}

