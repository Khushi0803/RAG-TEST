from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from pydantic import BaseModel
from main import load_retriever, build_and_save_vectorstore
from rag_pipeline import rag_simple, llm

app = FastAPI(title="RAG Retrieval API", docs_url="/", version="1.0.0")

retriever = None
retriever_ready = False

class QueryRequest(BaseModel):
    query: str

def init_retriever():
    global retriever, retriever_ready
    print("Rebuilding vector store with fastembed...")
    build_and_save_vectorstore()
    retriever = load_retriever()
    retriever_ready = True

init_retriever()

@app.get("/health")
def health():
    return {"status": "ok", "retriever_ready": retriever_ready}

@app.post("/rag")
def run_rag(request: QueryRequest):
    if not retriever_ready:
        return {"answer": "Model is loading, please retry in a few seconds."}
    answer = rag_simple(request.query, retriever, llm)
    return {"answer": answer}
