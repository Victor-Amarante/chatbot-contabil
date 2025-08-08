import os
import shutil
from pathlib import Path

import schemas
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

current_pdf_path = None
retrieval_chain = None
llm = None

def initialize_rag_system(pdf_path: str):
    """Inicializa o sistema RAG com um novo PDF"""
    global current_pdf_path, retrieval_chain, llm

    if llm is None:
        llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)

    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(
            chunk_size=1500, chunk_overlap=150, separator="\n"
        )
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever()

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "Responda a pergunta com base apenas no contexto abaixo:\n\n{context}\n"
                    "Pergunta: {input}"
                    "Caso não encontre informações suficientes, responda com 'Desculpe, não tenho informações suficientes para responder a essa pergunta.'"
                ),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )

        stuff_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=stuff_chain,
        )

        current_pdf_path = pdf_path
        return True
    except Exception as e:
        print(f"Erro ao inicializar RAG: {e}")
        return False


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
