from fastapi import APIRouter
from pydantic import BaseModel
from api.core.rag_service import chat_with_rag

router = APIRouter(prefix="/chat", tags=["Chat"])

class QueryRequest(BaseModel):
    query: str

@router.post("/")
def chat(request: QueryRequest):

    result = chat_with_rag(request.query)

    return {
        "query": request.query,
        "answer": result["answer"],
        "sources": result["sources"]
    }
