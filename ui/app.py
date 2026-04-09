import sys
import os
import streamlit as st

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from rag.pipeline import generate_answer

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("🤖 AI Research Assistant (RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------
# CHAT HISTORY
# -----------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# -----------------------
# INPUT
# -----------------------
user_input = st.chat_input("Ask something...")

if user_input:

    # user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # get response
    result = generate_answer(user_input)

    answer = result["answer"]
    sources = result["sources"]

    # -----------------------
    # ASSISTANT RESPONSE
    # -----------------------
    with st.chat_message("assistant"):

        st.markdown("### 🧠 Answer")
        st.markdown(answer)

        st.markdown("### 📚 Sources")

        if sources and len(sources) > 0:
            for i, s in enumerate(sources):
                st.markdown(f"**[{i+1}]** {s[:300]}")
        else:
            st.warning("No sources found for this query.")

    # save chat
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
