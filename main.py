from __future__ import annotations

import os
import sys
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# --- API Key ---
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    print("FATAL: API_KEY environment variable is not set. Exiting.")
    sys.exit(1)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/docs", "/openapi.json"):
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if not key or key != API_KEY:
            return Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)


# --- App ---
app = FastAPI(
    title="Qwen Summarization API",
    version="1.0.0",
    description="API for text summarization using Qwen 2.5. Single-request GPU lock.",
)
app.add_middleware(ApiKeyMiddleware)

# --- Config ---
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.9"))

GPU_LOCK = threading.Lock()
llm = None


@app.on_event("startup")
def load_model():
    global llm
    from vllm import LLM

    print(f"[*] Loading model: {MODEL_NAME} ...")
    start = time.time()
    llm = LLM(
        model=MODEL_NAME,
        quantization="awq",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        dtype="float16",
    )
    elapsed = time.time() - start
    print(f"[OK] Model loaded in {elapsed:.1f}s: {MODEL_NAME}")


# --- Health ---
@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    if llm is None:
        return PlainTextResponse("loading", status_code=503)
    return "ok"


# --- Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9


class SummarizeRequest(BaseModel):
    text: str
    language: str = "uk"
    max_tokens: int = 2048
    temperature: float = 0.3


# --- Endpoints ---
@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    if llm is None:
        return Response(
            content='{"detail":"Model is still loading"}',
            status_code=503,
            media_type="application/json",
        )

    from vllm import SamplingParams

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=req.top_p,
    )

    with GPU_LOCK:
        outputs = llm.chat(messages, sampling_params)

    output = outputs[0]
    text = output.outputs[0].text

    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "model": MODEL_NAME,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "total_tokens": len(output.prompt_token_ids) + len(output.outputs[0].token_ids),
        },
    }


SUMMARIZE_SYSTEM_PROMPT = """You are a lecture summarization assistant. Create a concise, structured summary of the provided lecture transcript.

The summary must include:
1. Main topic (one sentence)
2. Key points (bullet list)
3. Conclusions

Write in: {language}. Be concise but informative."""


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    if llm is None:
        return Response(
            content='{"detail":"Model is still loading"}',
            status_code=503,
            media_type="application/json",
        )

    from vllm import SamplingParams

    lang_map = {"uk": "Ukrainian", "ru": "Russian", "en": "English"}
    lang_name = lang_map.get(req.language, req.language)

    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT.format(language=lang_name)},
        {"role": "user", "content": req.text},
    ]

    sampling_params = SamplingParams(
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        top_p=0.9,
    )

    with GPU_LOCK:
        outputs = llm.chat(messages, sampling_params)

    output = outputs[0]
    text = output.outputs[0].text

    return {
        "summary": text,
        "model": MODEL_NAME,
        "usage": {
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run("main:app", host=host, port=port, workers=1)
