import os
from pathlib import Path
from typing import Annotated

import models
import schemas
from database import SessionLocal, engine
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
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
PDF_PATH = os.path.join(BASE_DIR, "data", "relatorio_anual_renner.pdf")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")

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

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

loader = PyPDFLoader(str(PDF_PATH))
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

@app.post("/question")
async def question(question: schemas.Question, db: Session = Depends(get_db)):
    """Endpoint para fazer perguntas sobre o PDF"""
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
