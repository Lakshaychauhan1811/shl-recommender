# SHL Assessment Recommender v2

## Stack
- FastAPI + Uvicorn
- Groq API (llama-3.3-70b-versatile) — FREE, fast
- 377 real SHL products from official catalog

## Quick Start

```bash
pip install -r requirements.txt
export GROQ_API_KEY="your-key-from-console.groq.com"
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Endpoints
- `GET /health` → `{"status": "ok"}`
- `POST /chat` → `{"messages": [...]}` → `{"reply":"...","recommendations":[...],"end_of_conversation":false}`

## Deploy to Render (free)
1. Push folder to GitHub
2. render.com → New Web Service → connect repo
3. Add env var: `GROQ_API_KEY`
4. Health check: `/health`

## Get free Groq API key
https://console.groq.com → Sign up → API Keys → Create key (free, no credit card)
