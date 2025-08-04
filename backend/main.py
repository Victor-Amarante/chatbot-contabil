import os
from pathlib import Path
from typing import Annotated
import shutil

import models
import schemas
from database import SessionLocal, engine
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


load_dotenv()

app = FastAPI()
models.Base.metadata.create_all(bind=engine)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Variáveis globais para armazenar o estado do RAG
current_pdf_path = None
retrieval_chain = None
llm = None

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
db_dependency = Annotated[Session, Depends(get_db)]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def initialize_rag_system(pdf_path: str):
    """Inicializa o sistema RAG com um novo PDF"""
    global current_pdf_path, retrieval_chain, llm
    
    if llm is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separator="\n")
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Responda a pergunta com base apenas no contexto abaixo:\n\n{context}\n"
                "Pergunta: {input}"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ])

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

@app.post("/upload-pdf", response_model=schemas.UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint para upload de PDF"""
    global current_pdf_path
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são permitidos")
    
    try:
        # Salva o arquivo no diretório data
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Inicializa o sistema RAG com o novo PDF
        if initialize_rag_system(file_path):
            return schemas.UploadResponse(
                message="PDF carregado com sucesso! Agora você pode fazer perguntas sobre o documento.",
                filename=file.filename
            )
        else:
            # Remove o arquivo se falhou ao inicializar
            os.remove(file_path)
            raise HTTPException(status_code=500, detail="Erro ao processar o PDF")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no upload: {str(e)}")

@app.post("/question")
async def question(question: schemas.Question, db: Session = Depends(get_db)):
    """Endpoint para fazer perguntas sobre o PDF"""
    global retrieval_chain, current_pdf_path
    
    if retrieval_chain is None or current_pdf_path is None:
        raise HTTPException(
            status_code=400, 
            detail="Nenhum PDF foi carregado. Faça o upload de um PDF primeiro."
        )
    
    try:
        result = await retrieval_chain.ainvoke({"input": question.question})
        answer = result["answer"] if isinstance(result, dict) else result
        db_q = models.QuestionBase(question=question.question, answer=answer)
        db.add(db_q)
        db.commit()
        db.refresh(db_q)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Endpoint para verificar o status do sistema"""
    return {
        "pdf_loaded": current_pdf_path is not None,
        "current_pdf": current_pdf_path,
        "rag_initialized": retrieval_chain is not None
    }
