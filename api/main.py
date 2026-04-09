from fastapi import FastAPI

from api.routes.chat import router as chat_router
from api.routes.search import router as search_router
from api.routes.ingest import router as ingest_router

app = FastAPI(title="Production RAG System")

app.include_router(chat_router)
app.include_router(search_router)
app.include_router(ingest_router)


@app.get("/")
def root():
    return {"status": "RAG API is running 🚀"}
