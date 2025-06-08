# LLM 응답 처리
from fastapi import APIRouter
from ufit.services.chat_service import get_llm_response

chat_router = APIRouter()

@chat_router.get("/chat")
def chat(prompt: str):
    response = get_llm_response(prompt)
    return {"response": response}