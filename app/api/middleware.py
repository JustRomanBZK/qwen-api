from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

OPEN_PATHS = ("/health", "/docs", "/openapi.json")


class ApiKeyMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        if request.url.path in OPEN_PATHS:
            return await call_next(request)
        key = request.headers.get("X-API-Key")
        if not key or key != self._api_key:
            return Response(
                content='{"detail":"Unauthorized"}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)
