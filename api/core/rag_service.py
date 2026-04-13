from rag.pipeline import generate_answer, retrieve_context

# Chat endpoint logic
def chat_with_rag(query: str):
    result = generate_answer(query)
    return result

# 🔥 FIX THIS (your error was here)
def search_only(query: str):
    contexts = retrieve_context(query)
    return {"results": contexts}
