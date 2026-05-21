"""estimate 기능만 단독으로 테스트하는 미니 서버."""
from fastapi import FastAPI
from routes import estimates

app = FastAPI(title="Estimate Test Server")
app.include_router(estimates.router)
