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
def expand_query(query: str):
    return [
        query,
        f"Explain {query} in NLP context",
        f"What is the concept of {query}",
        f"{query} deep learning explanation",
        f"{query} transformer attention relevance"
    ]

def deduplicate(docs):
    return list(dict.fromkeys(docs))


def compress_context(docs):
    return "\n".join([d[:300] for d in docs])

def rewrite_query(query: str) -> str:
    return f"NLP transformer attention information: {query}"


# ------------------------
# RETRIEVAL (HYBRID + RERANK)
# ------------------------
def retrieve_context(query):

    # 1. Multi-query expansion
    queries = expand_query(query)

    all_candidates = []

    # 2. Hybrid search for EACH query
    for q in queries:
        candidates = hybrid_search(
            queries=[q],
            faiss_index=faiss_index,
            bm25=bm25,
            chunks=chunks,
            top_k=10
        )
        all_candidates.extend(candidates)

    # 3. Deduplicate results
    candidates = deduplicate(all_candidates)

    # 4. Rerank (GPT-style ranking)
    top_docs = rerank(
        query=query,
        documents=candidates,
        top_k=6
    )

    return top_docs


# ------------------------
# PROMPT ENGINE (IMPORTANT)
# ------------------------
def build_prompt(query, contexts):

    return f"""
You are a GPT-4 level AI research assistant.

Your job:
- Answer ONLY using provided context
- Think step-by-step internally
- If information is missing, say:
  "I don't know based on provided documents"

CONTEXT:
{contexts}

QUESTION:
{query}

ANSWER:
Provide a clear, structured, technical explanation.
"""


# ------------------------
# MAIN GENERATION
# ------------------------
def generate_answer(query: str):

    contexts = retrieve_context(query)

    context_text = compress_context(contexts)

    prompt = build_prompt(query, context_text)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
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
