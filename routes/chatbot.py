from fastapi import APIRouter
from pydantic import BaseModel

from chatbot.chat_service import generate_chat_answer

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
def chat(req: ChatRequest):
    return generate_chat_answer(req.question)