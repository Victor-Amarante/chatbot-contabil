import os
import shutil
from pathlib import Path

import schemas
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from rag import initialize_rag_system

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-pdf", response_model=schemas.UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint para upload de PDF"""
    global current_pdf_path

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Apenas arquivos PDF são permitidos"
        )

    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if initialize_rag_system(file_path):
            return schemas.UploadResponse(
                message="PDF carregado com sucesso! Agora você pode fazer perguntas sobre o documento.",
                filename=file.filename,
            )
        else:
            os.remove(file_path)
            raise HTTPException(status_code=500, detail="Erro ao processar o PDF")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")


@app.post("/question")
async def question(question: schemas.Question):
    """Endpoint para fazer perguntas sobre o PDF"""
    global retrieval_chain, current_pdf_path

    if retrieval_chain is None or current_pdf_path is None:
        raise HTTPException(
            status_code=400,
            detail="Nenhum PDF foi carregado. Faça o upload de um PDF primeiro.",
        )

    try:
        result = await retrieval_chain.ainvoke({"input": question.question})
        answer = result["answer"] if isinstance(result, dict) else result
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Endpoint para verificar o status do sistema"""
    return {
        "pdf_loaded": current_pdf_path is not None,
        "current_pdf": current_pdf_path,
        "rag_initialized": retrieval_chain is not None,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
