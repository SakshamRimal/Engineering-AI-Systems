import streamlit as st
import requests
import os
import time

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")
st.title("🤖 AI Assistant")

# ── Sidebar ──────────────────────────────────────────────────────────────────
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
        if health.get("fallback"):
            st.info(f"Fallback: {health['fallback']}")
        cb_state = health.get("circuit_breaker", "closed")
        if cb_state == "open":
            st.warning(f"Circuit breaker: {cb_state}")
        cache_stats = health.get("cache", {})
        if cache_stats:
            st.caption(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.0%} ({cache_stats.get('size', 0)} entries)")
    except requests.exceptions.RequestException:
        st.error("Backend unreachable")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Re-ingest documents"):
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(f"{BACKEND_URL}/ingest", timeout=5)
                    if resp.ok:
                        st.success("Started in background")
                    else:
                        st.error(f"Failed: {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
    with col2:
        if st.button("Clear cache"):
            try:
                resp = requests.post(f"{BACKEND_URL}/cache/invalidate", timeout=5)
                if resp.ok:
                    st.success("Cache cleared")
            except requests.exceptions.RequestException:
                st.error("Failed")

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render existing conversation ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['document']} (chunk: {s['chunk_id']})")
        if msg.get("response_time"):
            st.caption(f"⏱ {msg['response_time']}ms")
        if msg.get("error"):
            st.error(msg["error"])

# ── Chat input ───────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

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
            start_time = time.time()
            try:
                resp = requests.post(
                    f"{BACKEND_URL}{endpoint}",
                    json={"message": user_input, "history": history},
                    timeout=90,
                )
                elapsed_ms = round((time.time() - start_time) * 1000)

                if resp.status_code == 429:
                    retry_after = resp.json().get("retry_after", 30)
                    error_msg = f"Rate limited. Retry after {retry_after}s"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": "(rate limited)",
                        "error": error_msg,
                    })
                elif resp.status_code == 503:
                    error_msg = "LLM service temporarily unavailable. Try again shortly."
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": "(unavailable)",
                        "error": error_msg,
                    })
                else:
                    resp.raise_for_status()
                    data = resp.json()

                    answer = data.get("answer", "(no answer returned)")
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                st.caption(f"📄 {s['document']} (chunk: {s['chunk_id']})")
                    st.caption(f"⏱ {elapsed_ms}ms")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "response_time": elapsed_ms,
                    })

            except requests.exceptions.ConnectionError:
                error_msg = "Cannot connect to backend. Is it running?"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": "(error)", "error": error_msg,
                })
            except requests.exceptions.Timeout:
                error_msg = "Request timed out. The model may be slow or overloaded."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": "(timeout)", "error": error_msg,
                })
            except requests.exceptions.RequestException as e:
                error_msg = f"Request failed: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": "(error)", "error": error_msg,
                })
