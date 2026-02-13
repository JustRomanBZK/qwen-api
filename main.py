from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import Config
from engine import LLMEngine
from middleware import ApiKeyMiddleware
from routes import create_router
from tasks import TaskStore

config = Config()
engine = LLMEngine(config)
task_store = TaskStore(ttl=config.task_ttl)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await engine.load()
    asyncio.create_task(task_store.cleanup_loop())
    yield


app = FastAPI(
    title="Qwen Summarization API",
    version="3.0.0",
    description="API for text summarization using Qwen 2.5 with async vLLM engine.",
    lifespan=lifespan,
)

app.add_middleware(ApiKeyMiddleware, api_key=config.api_key)
app.include_router(create_router(engine, task_store))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.host, port=config.port, workers=1)
