from fastapi import APIRouter
from api.core.rag_service import chat_with_rag

router = APIRouter()

@router.post("/chat")
def chat(query: str):
    return chat_with_rag(query)
