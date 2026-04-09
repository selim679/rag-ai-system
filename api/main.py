from fastapi import FastAPI
from api.routes import chat, search, ingest

app = FastAPI(
    title="RAG AI System",
    description="Production-ready RAG API (FAISS + BM25 + Rerank + LLM)",
    version="1.0"
)

app.include_router(chat.router)
app.include_router(search.router)
app.include_router(ingest.router)

@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG system is running 🚀"}
