from fastapi import FastAPI
from routes import estimates
from routes import chatbot

app = FastAPI(title="종합프로젝트 API")

app.include_router(estimates.router)
app.include_router(chatbot.router)