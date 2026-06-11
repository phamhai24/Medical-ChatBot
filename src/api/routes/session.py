"""Session history endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatHistoryResponse, ChatMessage
from src.core.redis_client import clear_chat_history, delete_session, generate_session_id, get_chat_history

router = APIRouter(prefix="/api/v1/chat", tags=["Session"])


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
def get_history(session_id: str):
    """Lấy lịch sử chat của một session."""
    messages = get_chat_history(session_id)

    return ChatHistoryResponse(
        session_id=session_id,
        messages=[ChatMessage(**msg) for msg in messages],
        total=len(messages),
    )


@router.delete("/history/{session_id}")
def delete_history(session_id: str):
    """Xóa lịch sử chat của một session."""
    delete_session(session_id)
    return {"success": True, "session_id": session_id, "message": "Session deleted"}


@router.post("/history/new")
def create_session():
    """Tạo một session ID mới."""
    new_id = generate_session_id()
    return {"session_id": new_id}
