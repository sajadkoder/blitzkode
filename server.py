#!/usr/bin/env python3
"""
BlitzKode - Optimized Backend Server
Creator: Sajad
"""

import os
import sys
from pathlib import Path
import llama_cpp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
)

print("Ready!\n")

# FastAPI app
app = FastAPI()
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

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    temperature = data.get("temperature", 0.2)
    max_tokens = data.get("max_tokens", 1024)
    
    full_prompt = f"{SYSTEM_PROMPT}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    result = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["<|im_end|>", "<|im_start|>user"],
    )
    
    response = result["choices"][0]["text"].strip()
    return JSONResponse({
        "response": response,
        "creator": "Sajad",
        "model": "BlitzKode"
    })

@app.get("/info")
async def info():
    return JSONResponse({
        "name": "BlitzKode",
        "creator": "Sajad",
        "version": "1.1",
        "status": "ready",
        "optimizations": ["35 GPU layers", "4096 ctx", "Flash Attention", "Memory locked"]
    })

if __name__ == "__main__":
    print("Open: http://localhost:7860")
    print("API: http://localhost:7860/info\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")
