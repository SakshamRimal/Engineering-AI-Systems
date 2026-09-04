import streamlit as st
import requests
import os

# Backend URL — configurable so this works both locally and in Docker Compose later
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")

# Sidebar: mode selection + backend status
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Response mode",
        ["Plain chat", "Chat with tools", "RAG (grounded + sources)"],
        index=2,
    )

    st.divider()
    st.subheader("Backend status")
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=3).json()
        st.success(f"Connected — provider: {health.get('provider', 'unknown')}")
    except requests.exceptions.RequestException:
        st.error("Backend unreachable")

    st.divider()
    if st.button("Re-ingest documents"):
        with st.spinner("Ingesting documents..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", timeout=120)
                if resp.ok:
                    st.success("Ingestion complete")
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['document']} (chunk: {s['chunk_id']})")

# Chat input
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build history payload from prior turns (excluding the message we just added)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]

    endpoint_map = {
        "Plain chat": "/chat",
        "Chat with tools": "/chat/tools",
        "RAG (grounded + sources)": "/chat/rag",
    }
    endpoint = endpoint_map[mode]

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json={"message": user_input, "history": history},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "(no answer returned)")
                sources = data.get("sources", [])

                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.caption(f"📄 {s['document']} (chunk: {s['chunk_id']})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.exceptions.RequestException as e:
                error_msg = f"Request to backend failed: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})