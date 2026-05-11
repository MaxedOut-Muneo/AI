from fastapi import FastAPI
from routes import estimates
from routes import chatbot
from routes import risk_detector

app = FastAPI(title="종합프로젝트 API")

app.include_router(estimates.router)
app.include_router(chatbot.router)
app.include_router(risk_detector.router)