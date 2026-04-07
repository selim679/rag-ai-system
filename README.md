# RAG AI System (End-to-End ML Pipeline)

A production-style **Retrieval-Augmented Generation (RAG) system** combining:

- FastAPI backend
- arXiv data ingestion pipeline
- FAISS vector database
- Airflow orchestration
- Streamlit UI
- Dockerized deployment

---

## 🧠 Architecture


---

## ⚙️ Tech Stack

- Python
- FastAPI
- LangChain
- OpenAI / LLMs
- FAISS
- Apache Airflow
- Streamlit
- Docker

---

## 📦 Features

✔ Automatic arXiv paper ingestion  
✔ ETL data pipeline  
✔ Vector embeddings storage  
✔ Semantic search (RAG)  
✔ REST API with FastAPI  
✔ UI dashboard (Streamlit)  
✔ Docker support  

---

## 🚀 How to Run

```bash
# install dependencies
pip install -r requirements.txt

# run API
uvicorn api.main:app --reload
