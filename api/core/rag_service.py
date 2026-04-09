from rag.pipeline import generate_answer, retrieve_context

def chat_with_rag(query: str):

    result = generate_answer(query)

    return {
        "answer": result["answer"],
        "sources": result["sources"]
    }


def search_only(query: str):

    return retrieve_context(query)
