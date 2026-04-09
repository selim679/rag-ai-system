from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def expand_query(query: str):
    prompt = f"""
Rewrite this question into 3 better search queries for a scientific paper database.

Question: {query}

Return format:
1.
2.
3.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    text = response.choices[0].message.content

    # extract queries
    lines = [l.strip("123. ").strip() for l in text.split("\n") if l.strip()]
    return [query] + lines[:3]
