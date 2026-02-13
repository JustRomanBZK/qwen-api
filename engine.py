import time
import uuid

from config import Config


class LLMEngine:

    def __init__(self, config: Config):
        self._config = config
        self._engine = None

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    async def load(self):
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        cfg = self._config
        print(f"[*] Loading model: {cfg.model_name} (max_len={cfg.max_model_len}) ...")
        start = time.time()

        engine_args = AsyncEngineArgs(
            model=cfg.model_name,
            quantization="awq_marlin",
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        elapsed = time.time() - start
        print(f"[OK] Model loaded in {elapsed:.1f}s: {cfg.model_name}")

    async def generate(self, messages: list[dict], temperature: float = 0.7,
                       max_tokens: int = 2048, top_p: float = 0.9) -> dict:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        request_id = str(uuid.uuid4())

        tokenizer = self._engine.get_tokenizer()
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        final_output = None
        async for output in self._engine.generate(
            prompt_text, sampling_params, request_id=request_id,
        ):
            final_output = output

        text = final_output.outputs[0].text

        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "model": self._config.model_name,
            "usage": {
                "prompt_tokens": len(final_output.prompt_token_ids),
                "completion_tokens": len(final_output.outputs[0].token_ids),
                "total_tokens": len(final_output.prompt_token_ids)
                + len(final_output.outputs[0].token_ids),
            },
        }
