import os
import sys


class Config:

    def __init__(self):
        self.api_key = os.environ.get("API_KEY")
        if not self.api_key:
            print("FATAL: API_KEY environment variable is not set. Exiting.")
            sys.exit(1)

        self.model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-32B-Instruct-AWQ")
        self.max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
        self.gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
        self.task_ttl = int(os.environ.get("TASK_TTL", "3600"))
        self.host = os.environ.get("HOST", "0.0.0.0")
        self.port = int(os.environ.get("PORT", "8001"))
