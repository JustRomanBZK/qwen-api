import asyncio
import time


class TaskStore:

    def __init__(self, ttl: int = 3600):
        self._tasks: dict[str, dict] = {}
        self._ttl = ttl

    def create(self, task_id: str):
        self._tasks[task_id] = {
            "status": "processing",
            "result": None,
            "error": None,
            "created": time.time(),
        }

    def get(self, task_id: str) -> dict | None:
        return self._tasks.get(task_id)

    def set_completed(self, task_id: str, result: dict):
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["result"] = result

    def set_failed(self, task_id: str, error: str):
        if task_id in self._tasks:
            self._tasks[task_id]["status"] = "failed"
            self._tasks[task_id]["error"] = error

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(300)
            now = time.time()
            expired = [k for k, v in self._tasks.items() if now - v["created"] > self._ttl]
            for k in expired:
                del self._tasks[k]
            if expired:
                print(f"[cleanup] Removed {len(expired)} expired tasks")
