# BlitzKode

<p align="center">
  <img src="Screenshot 2026-03-26 122611.png" alt="BlitzKode UI" width="800"/>
</p>

An AI-powered coding assistant built by **Sajad** using fine-tuned LLM technology.

## Features

- **Smart Code Generation** - Write clean, efficient, and well-documented code
- **Multiple Languages** - Python, JavaScript, Java, C++, and more
- **Fast Inference** - Optimized llama.cpp backend for rapid responses
- **Modern UI** - Clean, professional interface inspired by ChatGPT
- **Dark Theme** - Easy on the eyes for long coding sessions
- **Mobile Responsive** - Works on desktop and mobile devices
- **Streaming Support** - Real-time token streaming via API

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model | Qwen2.5-1.5B (fine-tuned) |
| Backend | Python + llama.cpp + FastAPI |
| Frontend | Vanilla HTML/CSS/JS |
| Training | HuggingFace Transformers + PEFT (LoRA) |

## Quick Start

### Run the Server

```bash
# Start the server
python server.py
```

Open **http://localhost:7860** in your browser.

### API Usage

```bash
# Generate code
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in python", "max_tokens": 200}'

# Stream response
curl -X POST http://localhost:7860/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write hello world in python"}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/generate` | POST | Generate response |
| `/generate/stream` | POST | Stream response |
| `/health` | GET | Health check |
| `/info` | GET | API info |

## Project Structure

```
BlitzKode/
├── server.py              # Main backend server
├── blitzkode.gguf         # Quantized model (3GB)
├── frontend/
│   └── index.html        # Web interface
├── Screenshot*.png        # UI screenshots
├── scripts/              # Training & export scripts
└── checkpoints/          # Trained model checkpoints
```

## System Prompt

BlitzKode is configured with the following traits:
- Expert in all programming languages
- Clean, efficient code with comments
- Concise explanations
- Practical solutions

## Performance

- **Model**: 1.5B parameters (quantized GGUF)
- **Context**: 2048 tokens
- **Mode**: CPU optimized inference

## Version

Current version: **1.0**

## Creator

Built with love by **Sajad**

## License

MIT License
