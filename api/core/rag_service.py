from rag.pipeline import generate_answer

# MAIN CHAT ENDPOINT
def chat_with_rag(query: str):
    return generate_answer(query)


# 🔥 ADD THIS (MISSING FUNCTION)
def search_only(query: str):
    result = generate_answer(query)

    return {
        "query": query,
        "sources": result["sources"]
    }
