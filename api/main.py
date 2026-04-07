from fastapi import FastAPI

app = FastAPI(title="RAG AI System")

@app.get("/")
def home():
    return {"message": "RAG System is running 🚀"}
