from dotenv import load_dotenv
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
