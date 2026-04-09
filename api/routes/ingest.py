from fastapi import APIRouter
from pydantic import BaseModel
from data_pipeline.arxiv_fetcher import fetch_arxiv, save_to_json

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

class IngestRequest(BaseModel):
    query: str
    max_results: int = 5

@router.post("/")
def ingest(request: IngestRequest):

    papers = fetch_arxiv(request.query, request.max_results)
    save_to_json(papers)

    return {
        "message": "Data ingested successfully",
        "count": len(papers)
    }
