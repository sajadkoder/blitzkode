#!/usr/bin/env python3
"""
BlitzKode - Optimized Backend Server
Creator: Sajad
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional
import llama_cpp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json

# Paths
BLITZKODE_DIR = Path("C:/Dev/Projects/BlitzKode")
MODEL_PATH = BLITZKODE_DIR / "blitzkode.gguf"
FRONTEND_PATH = BLITZKODE_DIR / "frontend" / "index.html"

if not MODEL_PATH.exists():
    print(f"ERROR: Model not found: {MODEL_PATH}")
    sys.exit(1)

print("=" * 50)
print("BLITZKODE")
print("Creator: Sajad")
print("=" * 50)

# Load model with maximum optimization
print("\nInitializing optimized engine...")
start_time = time.time()
llm = llama_cpp.Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=35,
    n_ctx=4096,
    n_threads=8,
    n_batch=512,
    verbose=False,
    use_mmap=True,
    use_mlock=True,
    flash_attn=True,
    tensor_split=[0.0],
    seed=-1,
)
load_time = time.time() - start_time
print(f"Model loaded in {load_time:.2f}s\n")

# FastAPI app
app = FastAPI(title="BlitzKode API", version="1.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Optimized system prompt - Creator attribution
SYSTEM_PROMPT = """<|im_start|>system
You are BlitzKode, an AI coding assistant created by Sajad.

Your traits:
- Expert in Python, JavaScript, Java, C++, and all programming languages
- Write clean, efficient, and well-documented code
- Provide concise explanations with code
- Always output working code when asked

Guidelines:
- Keep responses focused and practical
- Include comments in code
- Explain complex concepts simply
- Ask clarifying questions when needed<|im_end|>"""

@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_PATH))

@app.get("/health")
async def health():
    return JSONResponse({
        "status": "healthy",
        "model_loaded": True,
        "version": "1.2"
    })

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 0.3)
    max_tokens = data.get("max_tokens", 2048)
    top_p = data.get("top_p", 0.9)
    top_k = data.get("top_k", 40)
    repeat_penalty = data.get("repeat_penalty", 1.1)
    
    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    
    full_prompt = f"{SYSTEM_PROMPT}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    result = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["<|im_end|>", "<|im_start|>user"],
    )
    
    response = result["choices"][0]["text"].strip()
    return JSONResponse({
        "response": response,
        "creator": "Sajad",
        "model": "BlitzKode",
        "version": "1.2"
    })

@app.post("/generate/stream")
async def generate_stream(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 0.3)
    max_tokens = data.get("max_tokens", 2048)
    
    if not prompt:
        return JSONResponse({"error": "Prompt is required"}, status_code=400)
    
    full_prompt = f"{SYSTEM_PROMPT}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def generate_tokens():
        for token in llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["<|im_end|>", "<|im_start|>user"],
            stream=True,
        ):
            if token.get("choices"):
                text = token["choices"][0].get("text", "")
                if text:
                    yield f"data: {json.dumps({'token': text})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(
        generate_tokens(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@app.get("/info")
async def info():
    return JSONResponse({
        "name": "BlitzKode",
        "creator": "Sajad",
        "version": "1.2",
        "status": "ready",
        "optimizations": [
            "35 GPU layers",
            "4096 ctx",
            "Flash Attention",
            "Memory locked",
            "Streaming support",
            "Health endpoint"
        ],
        "endpoints": {
            "generate": "POST /generate",
            "stream": "POST /generate/stream",
            "health": "GET /health",
            "info": "GET /info"
        }
    })

if __name__ == "__main__":
    print("Open: http://localhost:7860")
    print("API: http://localhost:7860/info\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
