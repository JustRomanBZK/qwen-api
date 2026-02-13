import asyncio
import uuid

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, Response

from app.models.schemas import ChatRequest
from app.services.engine import LLMEngine
from app.services.task_store import TaskStore


def create_router(engine: LLMEngine, task_store: TaskStore) -> APIRouter:
    router = APIRouter()

    @router.get("/health", response_class=PlainTextResponse)
    def health() -> PlainTextResponse:
        if not engine.is_ready:
            return PlainTextResponse("loading", status_code=503)
        return PlainTextResponse("ok")

    @router.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        if not engine.is_ready:
            return Response(
                content='{"detail":"Model is still loading"}',
                status_code=503,
                media_type="application/json",
            )
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        return await engine.generate(messages, req.temperature, req.max_tokens, req.top_p)

    @router.post("/v1/tasks/create")
    async def create_task(req: ChatRequest):
        if not engine.is_ready:
            return Response(
                content='{"detail":"Model is still loading"}',
                status_code=503,
                media_type="application/json",
            )

        task_id = str(uuid.uuid4())
        task_store.create(task_id)
        asyncio.create_task(_run_task(task_id, req))
        return {"task_id": task_id, "status": "processing"}

    @router.get("/v1/tasks/{task_id}")
    async def get_task(task_id: str):
        task = task_store.get(task_id)
        if not task:
            return Response(
                content='{"detail":"Task not found"}',
                status_code=404,
                media_type="application/json",
            )

        resp = {"task_id": task_id, "status": task["status"]}
        if task["status"] == "completed":
            resp["result"] = task["result"]
        elif task["status"] == "failed":
            resp["error"] = task["error"]
        return resp

    async def _run_task(task_id: str, req: ChatRequest):
        try:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            result = await engine.generate(messages, req.temperature, req.max_tokens, req.top_p)
            task_store.set_completed(task_id, result)
        except Exception as e:
            task_store.set_failed(task_id, str(e))
            print(f"[task {task_id}] failed: {e}")

    return router
