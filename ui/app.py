import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

# ---------- Page config ----------
st.set_page_config(page_title="RAG DocQA Demo", page_icon="📄", layout="wide")

# ---------- Small helpers ----------
def card(title: str, icon: str = ""):
    """Simple visual section header."""
    st.markdown(
        f"""
        <div style="padding:14px 14px 4px 14px; border-radius:16px; border:1px solid rgba(120,120,120,0.25);">
            <h3 style="margin:0; padding:0;">{icon} {title}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

def spacer(h=8):
    st.markdown(f"<div style='height:{h}px'></div>", unsafe_allow_html=True)


# ---------- Header ----------
st.title("📄 RAG Document Q&A (Local Demo)")
st.caption("FastAPI + FAISS + Embeddings + Ollama • Upload a document → ask questions → get grounded answers.")
spacer(8)

# ---------- Sidebar: status + settings ----------
with st.sidebar:
    st.header("Backend")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200:
            st.success("API running ✅")
        else:
            st.warning(f"API responded: {r.status_code}")
    except Exception:
        st.error("API not reachable ❌")
        st.markdown("Start backend:")
        st.code("uvicorn app.main:app --reload")
        st.stop()

    st.divider()
    st.header("Settings")
    top_k = st.slider("Top-k retrieval", 1, 10, 4, help="How many chunks to retrieve before answering")
    timeout_s = st.slider("Request timeout (sec)", 30, 240, 120, step=10)

# ---------- Layout ----------
left, right = st.columns([1.05, 1.0], gap="large")

# Keep last answer in session state so it persists on reruns
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_ingest" not in st.session_state:
    st.session_state.last_ingest = None

# ---------- LEFT: Upload + Ask ----------
with left:
    card("1) Upload / Ingest Document", "⬆️")
    spacer(10)

    st.markdown("**Accepted formats:** `.txt`, `.pdf`")
    uploaded = st.file_uploader(
        "Choose a file",
        type=["txt", "pdf"],
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        ingest_btn = st.button("📥 Ingest Document", use_container_width=True, disabled=(uploaded is None))
    with col_b:
        clear_btn = st.button("🧹 Clear UI Output", use_container_width=True)

    if clear_btn:
        st.session_state.last_answer = None
        st.session_state.last_sources = []
        st.session_state.last_ingest = None
        st.success("Cleared UI output.")

    if ingest_btn and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        try:
            with st.spinner("Ingesting document..."):
                resp = requests.post(f"{API_BASE}/ingest", files=files, timeout=timeout_s)
            if resp.status_code == 200:
                st.session_state.last_ingest = resp.json()
                st.success("Document ingested ✅")
            else:
                st.error("Ingest failed ❌")
                st.code(resp.text)
        except Exception as e:
            st.error("Error calling /ingest")
            st.code(str(e))

    if st.session_state.last_ingest:
        st.markdown("**Last ingest result:**")
        st.json(st.session_state.last_ingest)

    spacer(18)
    card("2) Ask a Question", "❓")
    spacer(10)

    question = st.text_area(
        "Your question",
        placeholder="e.g., Summarize the key points. What does the document say about X?",
        height=110,
        label_visibility="collapsed",
    )

    ask_col1, ask_col2 = st.columns([1, 1])
    with ask_col1:
        ask_btn = st.button("🤖 Ask", use_container_width=True)
    with ask_col2:
        example_btn = st.button("✨ Use Example Question", use_container_width=True)

    if example_btn:
        st.session_state["example_q"] = "Summarize the document in 3 bullet points."
        st.rerun()

    if "example_q" in st.session_state and not question.strip():
        question = st.session_state["example_q"]

    if ask_btn:
        if not question.strip():
            st.warning("Please type a question first.")
        else:
            payload = {"question": question.strip(), "top_k": int(top_k)}
            try:
                with st.spinner("Retrieving context + generating answer..."):
                    resp = requests.post(f"{API_BASE}/ask", json=payload, timeout=timeout_s)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.last_answer = data.get("answer", "")
                    st.session_state.last_sources = data.get("sources", []) or []
                    st.success("Done ✅")
                else:
                    st.error("Ask failed ❌")
                    st.code(resp.text)
            except Exception as e:
                st.error("Error calling /ask")
                st.code(str(e))

# ---------- RIGHT: Answer + Sources ----------
with right:
    card("3) Answer", "✅")
    spacer(10)

    if st.session_state.last_answer:
        st.markdown("**Answer:**")
        st.text_area(
            "answer_box",
            value=st.session_state.last_answer,
            height=260,
            label_visibility="collapsed",
        )
    else:
        st.info("Upload a document and ask a question to see the answer here.")

    spacer(12)
    card("Sources", "📌")
    spacer(10)

    if st.session_state.last_sources:
        for s in st.session_state.last_sources:
            st.markdown(f"- `{s}`")
    else:
        st.caption("No sources yet (ingest a document and ask a question).")

    spacer(12)
    st.caption("Tip: If the answer is slow, use a smaller Ollama model like `phi3` and try a shorter PDF.")
