from fastapi import FastAPI
from .modules import insights_router

app = FastAPI(
    title="Strategic Intelligence API",
    description="Transforms raw JSON data payloads into strategic executive reports using CrewAI.",
    version="1.0.0",
)

app.include_router(insights_router, prefix="/api/v1")
