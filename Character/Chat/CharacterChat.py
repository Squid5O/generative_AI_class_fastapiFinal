from fastapi import APIRouter

router_Chat = APIRouter(prefix="/chat", tags=["chat"])

@router_Chat.get("/")
async def get_chat_history():
    # 채팅 내역 조회 로직
    return {"status": "Chat history retrieved successfully"}

@router_Chat.post("/")
async def send_message():
    # 메시지 전송 로직
    return {"status": "Message sent successfully"}
