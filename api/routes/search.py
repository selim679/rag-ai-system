from fastapi import APIRouter
from pydantic import BaseModel
from api.core.rag_service import search_only

router = APIRouter(prefix="/search", tags=["Search"])

class SearchRequest(BaseModel):
    query: str

@router.post("/")
def search(request: SearchRequest):

    results = search_only(request.query)

    return {
        "query": request.query,
        "results": results
    }
