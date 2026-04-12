from vector_db.auto_index import load_all
from vector_db.hybrid_search import hybrid_search
from vector_db.reranker import rerank
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
CHAT_MEMORY = []


def add_to_memory(role, message):
    CHAT_MEMORY.append({"role": role, "content": message})


def get_memory():
    return CHAT_MEMORY[-6:]  # last 6 messages

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

faiss_index, chunks, bm25 = load_all()


def expand_query(query):
    return [
        query,
        f"Explain {query} in NLP context",
        f"{query} transformer attention mechanism",
        f"{query} deep learning explanation",
        f"What is {query} in AI research"
    ]


def retrieve_context(query):

    queries = expand_query(query)

    candidates = hybrid_search(
        queries=queries,
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        top_k=15
    )

    return rerank(query, candidates, top_k=6)


def build_prompt(query, contexts):

    memory_text = "\n".join(
        [f"{m['role']}: {m['content']}" for m in get_memory()]
    )

    context_text = "\n\n".join(contexts)

    return f"""
You are ChatGPT-level AI assistant.

Conversation memory:
{memory_text}

Context:
{context_text}

User question:
{query}

Rules:
- Use memory + context
- Be natural like ChatGPT
- If unsure say "I don't know"

Answer:
"""


def generate_answer(query: str):

    add_to_memory("user", query)

    contexts = retrieve_context(query)

    prompt = build_prompt(query, contexts)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content

    add_to_memory("assistant", answer)

    return {
        "answer": answer,
        "sources": contexts
    }
def generate_stream(query: str):

    result = generate_answer(query)
    answer = result["answer"]

    for token in answer.split():
        yield token + " "