from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import aiofiles

from .schemas import AskRequest, AskResponse
from .ingest import load_text_file, load_pdf_file, chunk_text
from .rag import load_or_create_store, add_documents, retrieve, answer_with_ollama

app = FastAPI(title="RAG DocQA Demo (Ollama)", version="1.0.0")

store = load_or_create_store()

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not (filename.endswith(".txt") or filename.endswith(".pdf")):
        raise HTTPException(
            status_code=400,
            detail="Upload a .txt or .pdf file."
        )

    save_path = UPLOAD_DIR / file.filename

    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    if filename.endswith(".pdf"):
        text = load_pdf_file(save_path)
    else:
        text = load_text_file(save_path)
    docs = chunk_text(text, source_name=file.filename)
    add_documents(store, docs)

    return {
        "message": "Ingested",
        "file": file.filename,
        "chunks_added": len(docs),
    }

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    docs = retrieve(store, payload.question, k=payload.top_k)
    answer = answer_with_ollama(payload.question, docs)
    sources = sorted({d.metadata.get("source", "unknown") for d in docs})
    return AskResponse(answer=answer, sources=sources)
