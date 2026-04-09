from vector_db.auto_index import load_all
from vector_db.hybrid_search import hybrid_search
from vector_db.reranker import rerank

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# AUTO LOAD SYSTEM
faiss_index, chunks, bm25 = load_all()


def retrieve_context(query):

    # 1. hybrid search
    candidates = hybrid_search(query, faiss_index, bm25, chunks, top_k=10)

    # 2. rerank
    final_docs = rerank(query, candidates, top_k=3)

    return final_docs


def generate_answer(query):

    contexts = retrieve_context(query)

    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an AI research assistant.

Answer ONLY using context.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": contexts
    }
