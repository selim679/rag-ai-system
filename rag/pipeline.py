from vector_db.faiss_index import load_index, load_chunks, semantic_search
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

index = load_index()
chunks = load_chunks()

def retrieve_context(query):
    return semantic_search(query, index, chunks, top_k=3)

def generate_answer(query):

    contexts = retrieve_context(query)
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are an AI assistant.

Use ONLY the context below.

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
