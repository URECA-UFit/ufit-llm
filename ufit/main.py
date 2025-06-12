from fastapi import FastAPI
from ufit.routes.chat import chat_router  # 예: ufit/routes/sample.py 에서 가져오는 라우터
from ufit.routes.recommend_routes import recommend_router

app = FastAPI()

app.include_router(chat_router)
app.include_router(recommend_router)

@app.get("/")
def read_root():
    return {"message": "ufit LLM 서버 작동 중!"}