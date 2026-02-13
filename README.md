# Qwen 2.5 Summarization API

HTTP API for text summarization using Qwen 2.5-32B (AWQ quantized) with X-API-Key auth and GPU lock.

## Quick Start (Ubuntu + NVIDIA GPU)

```bash
git clone https://github.com/JustRomanBZK/qwen-api.git
cd qwen-api
chmod +x setup.sh
./setup.sh
```

Script auto-installs: NVIDIA driver, Docker, NVIDIA Container Toolkit, creates `.env`, builds and starts the service.

**First launch downloads ~20GB model. Takes 10-30 min depending on connection.**

## Manual Deploy

```bash
git clone https://github.com/JustRomanBZK/qwen-api.git
cd qwen-api
cp .env.example .env     # set API_KEY
docker compose up -d
```

Docs: `http://localhost:8001/docs`

## Auth

All requests (except `/health`) require `X-API-Key` header:

```bash
curl -H "X-API-Key: your-key" http://localhost:8001/health
```

## API

### GET /health

No auth. Returns `ok` (200) when ready, `loading` (503) while model loads.

### POST /summarize

Convenience endpoint for lecture summarization.

```bash
curl -X POST http://localhost:8001/summarize \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Full lecture transcript text here...",
    "language": "uk",
    "max_tokens": 2048
  }'
```

Response:
```json
{
  "summary": "## Main topic\n...\n## Key points\n- ...",
  "model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
  "usage": {"prompt_tokens": 1234, "completion_tokens": 256}
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Transcript to summarize |
| `language` | string | `uk` | Output language (uk/ru/en) |
| `max_tokens` | int | 2048 | Max response tokens |
| `temperature` | float | 0.3 | Creativity (0.0 - 1.0) |

### POST /v1/chat/completions

OpenAI-compatible chat endpoint for custom prompts.

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize this text..."}
    ],
    "temperature": 0.7,
    "max_tokens": 2048
  }'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | — | **Required.** Auth key |
| `HF_TOKEN` | — | HuggingFace token (optional) |
| `MODEL_NAME` | `Qwen/Qwen2.5-32B-Instruct-AWQ` | Model to load |
| `MAX_MODEL_LEN` | `8192` | Max context length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | VRAM usage (0.0 - 1.0) |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8001` | Bind port |

## Hardware Requirements

- NVIDIA GPU with 48GB+ VRAM (A6000, A100, etc.)
- 32GB+ RAM
- 50GB+ disk for model cache

## Notes

- Single request at a time (GPU lock) — suitable for batch summarization
- Model cached in Docker volume `hf-cache` — survives container restarts
- AWQ 4-bit quantization: ~20GB VRAM, fast inference
