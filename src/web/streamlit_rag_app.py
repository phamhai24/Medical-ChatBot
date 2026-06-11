"""Streamlit web UI for Medical Chatbot RAG"""

import streamlit as st
import requests
import time
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = "http://localhost:8000"
WEB_CONFIG = {
    "title": "Chatbot Y tế - RAG",
    "icon": "🏥",
    "description": "Trợ lý y tế sử dụng Retrieval-Augmented Generation",
    "page_layout": "wide",
}

# ─── API Helpers ────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_chat_api(message: str, top_k: int = 5, temperature: Optional[float] = None) -> dict:
    """Call the chat API."""
    resp = requests.post(
        f"{API_BASE_URL}/chat",
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


def call_ingest_api(data_path: str = "data/processed/data.json", rebuild: bool = False) -> dict:
    """Call the ingest API."""
    resp = requests.post(
        f"{API_BASE_URL}/ingest",
        json={
            "data_path": data_path,
            "rebuild": rebuild,
            "batch_size": 100,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=WEB_CONFIG["title"],
    page_icon=WEB_CONFIG["icon"],
    layout=WEB_CONFIG["page_layout"],
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Cấu hình")

    # API connection status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("🟢 API đang hoạt động")
    else:
        st.error("🔴 API không kết nối được")
        st.info(f"Khởi động API: `python -m src.api.server`\n(API chạy tại {API_BASE_URL})")

    st.divider()

    # Retrieval settings
    st.markdown("### 🔍 Tìm kiếm")
    top_k = st.slider(
        "Số tài liệu tìm kiếm (top_k):",
        min_value=1,
        max_value=20,
        value=5,
        help="Số lượng tài liệu được tìm và đưa vào context"
    )

    st.divider()

    # Generation settings
    st.markdown("### ✍️ Sinh text")
    temperature = st.slider(
        "Nhiệt độ (Temperature):",
        min_value=0.0,
        max_value=1.5,
        value=0.3,
        step=0.05,
        help="Thấp = bảo thủ, Cao = sáng tạo"
    )

    st.divider()

    # Admin actions
    st.markdown("### 🗄️ Quản lý dữ liệu")

    if st.button("📥 Ingest dữ liệu", use_container_width=True):
        with st.spinner("Đang ingest dữ liệu..."):
            try:
                result = call_ingest_api(rebuild=True)
                st.success(
                    f"✅ Ingest thành công!\n"
                    f"- Records: {result['stats'].get('total_records', 'N/A')}\n"
                    f"- Chunks: {result['stats'].get('total_chunks', 'N/A')}\n"
                    f"- Docs indexed: {result['stats'].get('documents_indexed', 'N/A')}"
                )
            except Exception as e:
                st.error(f"❌ Lỗi ingest: {e}")

    if st.button("🔄 Rebuild Index", use_container_width=True):
        with st.spinner("Đang rebuild..."):
            try:
                result = call_ingest_api(rebuild=True)
                st.success(f"✅ Rebuild thành công! {result['stats'].get('documents_indexed', 0)} docs")
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

    st.divider()

    # Stats
    st.markdown("### 📊 Thống kê")
    if st.button("📡 Kiểm tra stats", use_container_width=True):
        try:
            resp = requests.get(f"{API_BASE_URL}/stats", timeout=5)
            stats = resp.json()
            st.json(stats)
        except Exception as e:
            st.error(f"Lỗi: {e}")

    st.divider()
    st.caption("⚠️ **Lưu ý:** Chatbot demo giáo dục. Không dùng cho chẩn đoán y khoa.")


# ─── Main Content ─────────────────────────────────────────────────────────────

st.title(f"{WEB_CONFIG['icon']} {WEB_CONFIG['title']}")
st.markdown(f"*{WEB_CONFIG['description']}*")

# Model info banner
if api_healthy:
    try:
        resp = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        stats = resp.json()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Embedding", stats.get("embedding_model", "N/A").split("/")[-1])
        with col2:
            st.metric("Vector Store", stats.get("vector_store_type", "N/A").upper())
        with col3:
            st.metric("Documents", stats.get("document_count", 0))
        with col4:
            st.metric("Generator", stats.get("generation_model", "N/A").split("/")[-1])
    except Exception:
        pass

st.divider()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = 0

# Display chat history
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🏥"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Nguồn trích dẫn"):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(f"**[{i+1}]** {src.get('question', 'N/A')} — score: `{src.get('score', 0):.4f}`")
        if msg.get("latency"):
            st.caption(f"⏱️ {msg['latency']}ms")

# Chat input
if prompt := st.chat_input("Hỏi về sức khỏe, bệnh tật, thuốc men..."):
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "sources": None,
        "latency": None,
    })

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant", avatar="🏥"):
        if not api_healthy:
            st.error(
                "⚠️ API không hoạt động. Vui lòng khởi động API trước:\n\n"
                "```bash\npython -m src.api.server\n```"
            )
        else:
            # Streaming placeholder
            response_placeholder = st.empty()

            try:
                start = time.time()
                result = call_chat_api(
                    message=prompt,
                    top_k=top_k,
                    temperature=temperature,
                )
                elapsed = time.time() - start

                # Display answer
                response_placeholder.markdown(result["answer"])

                # Display sources
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"📚 Nguồn trích dẫn ({len(sources)} results)"):
                        for i, src in enumerate(sources):
                            score = src.get("score", 0)
                            # Convert distance to similarity (approximate)
                            similarity = max(0, min(1, 1 - score)) if score else 0
                            st.markdown(
                                f"**[{i+1}]** {src.get('question', 'N/A')}\n"
                                f"_Similarity: {similarity:.2f} | Latency: {result.get('latency_ms', 0)}ms_"
                            )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": sources,
                    "latency": f"{result.get('latency_ms', round(elapsed * 1000, 1))}",
                })

            except requests.exceptions.HTTPError as e:
                error_msg = f"❌ Lỗi HTTP: {e}"
                if e.response.status_code == 500:
                    error_msg = "❌ Lỗi server. Vui lòng thử lại hoặc kiểm tra log."
                response_placeholder.error(error_msg)
                logger.error(f"Chat API error: {e}")

            except Exception as e:
                error_msg = f"❌ Lỗi: {str(e)}"
                response_placeholder.error(error_msg)
                logger.error(f"Chat error: {e}")

# Clear chat button
if st.session_state.messages and st.button("🗑️ Xóa lịch sử chat"):
    st.session_state.messages = []
    st.rerun()
