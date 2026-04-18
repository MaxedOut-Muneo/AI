from fastapi import FastAPI
from routes import estimates

app = FastAPI(title="종합프로젝트 API")

app.include_router(estimates.router)
