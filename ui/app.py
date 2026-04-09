import streamlit as st
from rag.pipeline import generate_answer

st.set_page_config(page_title="ChatGPT RAG", page_icon="🤖")

st.title("💬 ChatGPT-style RAG Assistant")

# session state memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input box
user_input = st.chat_input("Ask me anything...")

if user_input:

    # user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant response placeholder
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        result = generate_answer(user_input)
        answer = result["answer"]
        sources = result["sources"]

        # STREAMING EFFECT
        for word in answer.split():
            full_response += word + " "
            placeholder.markdown(full_response)

    # save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    # show sources
    with st.expander("📚 Sources"):
        for i, s in enumerate(sources):
            st.write(f"{i+1}. {s[:250]}")
