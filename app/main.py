from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .modules import insights_router

app = FastAPI(
    title="Insights Intelligence API",
    description="Transforms raw JSON data payloads into strategic analysis reports.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(insights_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
