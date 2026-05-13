from fastapi import APIRouter
from chatbot.chat_service import ChatbotService
from chatbot.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/chatbot", tags=["chatbot"])
service = ChatbotService()

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    return service.chat(req.question)