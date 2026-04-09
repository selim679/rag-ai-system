import os
from dotenv import load_dotenv
from openai import OpenAI

from vector_db.auto_index import load_all
from vector_db.hybrid_search import hybrid_search
from vector_db.reranker import rerank

load_dotenv()

# ------------------------
# LLM CLIENT (GROQ)
# ------------------------
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# ------------------------
# LOAD DATA ONCE
# ------------------------
faiss_index, chunks, bm25 = load_all()


# ------------------------
# QUERY REWRITE (optional upgrade)
# ------------------------
def rewrite_query(query: str) -> str:
    return f"NLP transformer attention information: {query}"


# ------------------------
# RETRIEVAL (HYBRID + RERANK)
# ------------------------
def retrieve_context(query: str):

    query = rewrite_query(query)

    # 1. Hybrid search
    candidates = hybrid_search(
        queries=[query],      # IMPORTANT: list
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        top_k=20
    )

    # 2. Reranking (Cross-Encoder)
    top_docs = rerank(
        query=query,
        documents=candidates,
        top_k=6
    )

    return top_docs


# ------------------------
# PROMPT ENGINE (IMPORTANT)
# ------------------------
def build_prompt(query: str, contexts: list):

    context_text = "\n\n".join(contexts)

    return f"""
You are a ChatGPT-level AI research assistant.

RULES:
- Use ONLY the provided context
- If the answer is not in the context, say:
  "I don't know based on the provided documents"
- Do NOT use external knowledge
- Be precise, technical, and structured

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:
"""


# ------------------------
# MAIN GENERATION
# ------------------------
def generate_answer(query: str):

    contexts = retrieve_context(query)
    prompt = build_prompt(query, contexts)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": contexts
    }


# ------------------------
# STREAM (Streamlit UI)
# ------------------------
def generate_stream(query: str):

    result = generate_answer(query)
    answer = result["answer"]

    for word in answer.split():
        yield word + " "
