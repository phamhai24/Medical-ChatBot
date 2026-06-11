"""Streamlit web UI - refactored with modular components."""

import os
import time

import streamlit as st

# ─── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
WEB_CONFIG = {
    "title": "Medical RAG Chatbot",
    "icon": "🏥",
    "description": "Trợ lý y tế sử dụng Retrieval-Augmented Generation",
    "layout": "wide",
}

# ─── API Helpers ───────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    try:
        import requests
        resp = requests.get(f"{API_BASE_URL}/", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def call_chat(message: str, top_k: int = 5, temperature: float = 0.3) -> dict:
    import requests
    resp = requests.post(
        f"{API_BASE_URL}/api/v1/chat/ask",
        json={
            "message": message,
            "top_k": top_k,
            "temperature": temperature,
            "include_sources": True,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_ingest(rebuild: bool = False) -> dict:
    import requests
    resp = requests.post(
        f"{API_BASE_URL}/api/v1/admin/ingest",
        json={"rebuild": rebuild, "batch_size": 100},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def call_stats() -> dict:
    import requests
    resp = requests.get(f"{API_BASE_URL}/stats", timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─── Streamlit Setup ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title=WEB_CONFIG["title"],
    page_icon=WEB_CONFIG["icon"],
    layout=WEB_CONFIG["layout"],
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-ok { color: #28a745; }
    .status-error { color: #dc3545; }
    .stSidebar .stMarkdown h1 { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title(f"{WEB_CONFIG['icon']} Cấu hình")

    # API status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("🟢 API đang hoạt động")
    else:
        st.error("🔴 API không kết nối")
        st.info(f"Khởi động API:\n```bash\npython -m src.api.main\n```")

    st.divider()

    # Retrieval settings
    st.markdown("### 🔍 Retrieval")
    top_k = st.slider("Số tài liệu (top_k)", 1, 20, 5)
    score_threshold = st.slider("Ngưỡng similarity", 0.0, 1.0, 0.3, 0.05)

    st.divider()

    # Generation settings
    st.markdown("### ✍️ Generation")
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05)

    st.divider()

    # Admin
    st.markdown("### 🗄️ Quản lý")

    if st.button("📥 Ingest dữ liệu", use_container_width=True):
        with st.spinner("Đang ingest..."):
            try:
                result = call_ingest(rebuild=True)
                st.success(
                    f"✅ Thành công!\n"
                    f"- Records: {result.get('total_records', 'N/A')}\n"
                    f"- Chunks: {result.get('total_chunks', 'N/A')}\n"
                    f"- Thời gian: {result.get('duration_seconds', 'N/A')}s"
                )
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    st.divider()

    # Stats
    st.markdown("### 📊 Thống kê")
    if st.button("📡 Cập nhật stats", use_container_width=True):
        try:
            stats = call_stats()
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Documents", stats.get("document_count", 0))
            with col_b:
                st.metric("Vector DB", stats.get("vector_store_type", "N/A").upper())
        except Exception:
            st.warning("Không lấy được stats")

    st.divider()
    st.caption("⚠️ Chatbot demo giáo dục. Không dùng cho chẩn đoán y khoa.")


# ─── Main Content ──────────────────────────────────────────────────────────────

st.title(f"{WEB_CONFIG['icon']} {WEB_CONFIG['title']}")
st.markdown(WEB_CONFIG['description'])

# Model info banner
if api_healthy:
    try:
        stats = call_stats()
        cols = st.columns(5)
        with cols[0]:
            emb_model = stats.get("embedding_model", "N/A")
            st.caption("Embedding")
            st.code(emb_model.split("/")[-1] if emb_model else "N/A", language=None)
        with cols[1]:
            gen_model = stats.get("generation_model", "N/A")
            st.caption("Generator")
            st.code(gen_model.split("/")[-1] if gen_model else "N/A", language=None)
        with cols[2]:
            st.caption("Vector Store")
            st.code(stats.get("vector_store_type", "N/A").upper(), language=None)
        with cols[3]:
            st.caption("Documents")
            st.code(str(stats.get("document_count", 0)), language=None)
        with cols[4]:
            st.caption("Top K")
            st.code(str(stats.get("retrieval_top_k", 5)), language=None)
    except Exception:
        pass

st.divider()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Display chat history
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🏥"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Nguồn ({len(msg['sources'])} results)"):
                for i, src in enumerate(msg["sources"]):
                    score = src.get("score", 0)
                    similarity = max(0, min(1, 1 - score)) if score else 0
                    st.markdown(
                        f"**[{i+1}]** {src.get('question', 'N/A')}\n"
                        f"_{similarity:.2f} similarity_"
                    )
        if msg.get("latency"):
            st.caption(f"⏱️ {msg['latency']}ms")

# Chat input
if prompt := st.chat_input("Hỏi về sức khỏe, bệnh tật, thuốc men..."):
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": None,
        "latency": None,
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant", avatar="🏥"):
        if not api_healthy:
            st.error(
                "⚠️ API không hoạt động. Vui lòng khởi động API trước:\n\n"
                "```bash\npython -m src.api.main\n```"
            )
        else:
            try:
                start = time.perf_counter()
                result = call_chat(prompt, top_k=top_k, temperature=temperature)
                elapsed = time.time() - start

                st.markdown(result["answer"])

                # Sources
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 Nguồn ({len(sources)} results)"):
                        for i, src in enumerate(sources):
                            score = src.get("score", 0)
                            similarity = max(0, min(1, 1 - score)) if score else 0
                            st.markdown(
                                f"**[{i+1}]** {src.get('question', 'N/A')}\n"
                                f"_{similarity:.2f} similarity_"
                            )

                st.caption(f"⏱️ {result.get('latency_ms', round(elapsed * 1000, 1))}ms")

                # Save
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "latency": str(result.get('latency_ms', round(elapsed * 1000, 1))),
                })

            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")

# Clear button
if st.session_state.messages and st.button("🗑️ Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()
