#!/usr/bin/env python3
"""
BlitzKode backend server.

The API serves the bundled frontend and proxies prompts to a local GGUF model
through llama.cpp. The model is loaded lazily so the module stays importable in
tests and in environments where the model artifact is not present yet.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Iterator

import llama_cpp
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

APP_NAME = "BlitzKode"
APP_VERSION = "1.6"
CREATOR = "Sajad"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "blitzkode.gguf"
DEFAULT_FRONTEND_PATH = ROOT_DIR / "frontend" / "index.html"
DEFAULT_CONTEXT = 2048
DEFAULT_MAX_PROMPT_LENGTH = 4000
DEFAULT_MAX_TOKENS = 512

SYSTEM_PROMPT = """<|im_start|>system
You are BlitzKode, an AI coding assistant created by Sajad. You are an expert in Python, JavaScript, Java, C++, and other programming languages. Write clean, efficient, and well-documented code. Keep responses concise and practical.<|im_end|>"""


def _bool_from_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    root_dir: Path = ROOT_DIR
    model_path: Path = Path(os.getenv("BLITZKODE_MODEL_PATH", DEFAULT_MODEL_PATH))
    frontend_path: Path = Path(os.getenv("BLITZKODE_FRONTEND_PATH", DEFAULT_FRONTEND_PATH))
    host: str = os.getenv("BLITZKODE_HOST", "0.0.0.0")
    port: int = _int_from_env("BLITZKODE_PORT", 7860)
    n_ctx: int = _int_from_env("BLITZKODE_N_CTX", DEFAULT_CONTEXT)
    n_threads: int = _int_from_env(
        "BLITZKODE_THREADS",
        max(1, min(8, os.cpu_count() or 1)),
    )
    n_batch: int = _int_from_env("BLITZKODE_BATCH", 128)
    max_prompt_length: int = _int_from_env("BLITZKODE_MAX_PROMPT_LENGTH", DEFAULT_MAX_PROMPT_LENGTH)
    preload_model: bool = _bool_from_env("BLITZKODE_PRELOAD_MODEL", default=False)


class GenerateRequest(BaseModel):
    prompt: str
    temperature: float = Field(default=0.5, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=DEFAULT_MAX_TOKENS)
    top_p: float = Field(default=0.95, gt=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=200)
    repeat_penalty: float = Field(default=1.05, ge=0.8, le=2.0)


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._llm = None
        self._lock = Lock()
        self._load_time_seconds: float | None = None
        self._last_error: str | None = None

    @property
    def model_loaded(self) -> bool:
        return self._llm is not None

    @property
    def model_exists(self) -> bool:
        return self.settings.model_path.exists()

    @property
    def last_error(self) -> str | None:
        return self._last_error

    @property
    def load_time_seconds(self) -> float | None:
        return self._load_time_seconds

    def load_model(self):
        if self._llm is not None:
            return self._llm

        with self._lock:
            if self._llm is not None:
                return self._llm

            if not self.model_exists:
                self._last_error = f"Model not found at {self.settings.model_path}"
                raise FileNotFoundError(self._last_error)

            start_time = time.perf_counter()
            try:
                self._llm = llama_cpp.Llama(
                    model_path=str(self.settings.model_path),
                    n_gpu_layers=0,
                    n_ctx=self.settings.n_ctx,
                    n_threads=self.settings.n_threads,
                    n_batch=self.settings.n_batch,
                    verbose=False,
                    use_mmap=True,
                    use_mlock=False,
                    seed=-1,
                )
                self._load_time_seconds = time.perf_counter() - start_time
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
                raise

        return self._llm

    def build_prompt(self, prompt: str) -> str:
        return f"{SYSTEM_PROMPT}<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def generate_once(self, req: GenerateRequest) -> dict[str, object]:
        llm = self.load_model()
        result = llm(
            self.build_prompt(req.prompt),
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repeat_penalty=req.repeat_penalty,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<|im_end|>", "<|im_start|>user"],
        )
        response = result["choices"][0]["text"].strip()
        return {
            "response": response,
            "creator": CREATOR,
            "model": APP_NAME,
            "version": APP_VERSION,
        }

    def stream_tokens(self, req: GenerateRequest) -> Iterator[str]:
        llm = self.load_model()
        try:
            for token in llm(
                self.build_prompt(req.prompt),
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repeat_penalty=req.repeat_penalty,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["<|im_end|>", "<|im_start|>user"],
                stream=True,
            ):
                if not token.get("choices"):
                    continue
                text = token["choices"][0].get("text", "")
                if text:
                    yield f"data: {json.dumps({'token': text})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    model_service = ModelService(settings)
    executor = ThreadPoolExecutor(max_workers=1)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.preload_model:
            try:
                await asyncio.to_thread(model_service.load_model)
            except Exception:
                # Health and generation endpoints expose the error more clearly.
                pass

        try:
            yield
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title=f"{APP_NAME} API", version=APP_VERSION, lifespan=lifespan)
    app.state.settings = settings
    app.state.model_service = model_service
    app.state.executor = executor

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        if not settings.frontend_path.exists():
            raise HTTPException(status_code=404, detail="Frontend file is missing.")
        return FileResponse(str(settings.frontend_path))

    @app.get("/health")
    async def health():
        status = "healthy"
        if not settings.frontend_path.exists() or not model_service.model_exists:
            status = "degraded"

        return JSONResponse(
            {
                "status": status,
                "model_loaded": model_service.model_loaded,
                "model_path": str(settings.model_path),
                "model_exists": model_service.model_exists,
                "frontend_exists": settings.frontend_path.exists(),
                "version": APP_VERSION,
                "last_error": model_service.last_error,
            }
        )

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        prompt = req.prompt.strip()
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        if len(prompt) > settings.max_prompt_length:
            return JSONResponse(
                {"error": f"Prompt too long. Max {settings.max_prompt_length} chars."},
                status_code=400,
            )

        sanitized_request = req.model_copy(update={"prompt": prompt})

        try:
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(
                executor,
                model_service.generate_once,
                sanitized_request,
            )
            return JSONResponse(payload)
        except FileNotFoundError as exc:
            return JSONResponse({"error": str(exc)}, status_code=503)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest):
        prompt = req.prompt.strip()
        if not prompt:
            return JSONResponse({"error": "Prompt is required"}, status_code=400)

        if len(prompt) > settings.max_prompt_length:
            return JSONResponse(
                {"error": f"Prompt too long. Max {settings.max_prompt_length} chars."},
                status_code=400,
            )

        sanitized_request = req.model_copy(update={"prompt": prompt})

        if not model_service.model_exists:
            return JSONResponse(
                {"error": f"Model not found at {settings.model_path}"},
                status_code=503,
            )

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            model_service.stream_tokens(sanitized_request),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.get("/info")
    async def info():
        return JSONResponse(
            {
                "name": APP_NAME,
                "creator": CREATOR,
                "version": APP_VERSION,
                "status": "ready" if model_service.model_exists else "model-missing",
                "mode": "CPU (llama.cpp)",
                "context_window": settings.n_ctx,
                "model_loaded": model_service.model_loaded,
                "load_time_seconds": model_service.load_time_seconds,
                "endpoints": {
                    "generate": "POST /generate",
                    "stream": "POST /generate/stream",
                    "health": "GET /health",
                    "info": "GET /info",
                },
            }
        )

    return app


app = create_app()


def main() -> None:
    settings = app.state.settings
    print("=" * 50)
    print(APP_NAME.upper())
    print(f"Creator: {CREATOR}")
    print("=" * 50)
    print(f"Frontend: {settings.frontend_path}")
    print(f"Model: {settings.model_path}")
    print(f"Open: http://localhost:{settings.port}")
    print(f"API: http://localhost:{settings.port}/info")

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="warning")


if __name__ == "__main__":
    main()
