# Chat With PDF - FastAPI Backend

A lightweight FastAPI backend using HuggingFace Flan-T5-Small for PDF Q&A.

## Deploy to Railway

1. Create a [Railway](https://railway.app) account
2. Click "New Project" → "Deploy from GitHub repo"
3. Select this repository and set the root directory to `/backend`
4. Railway will auto-deploy!

## Environment Variables

No API keys needed - uses free HuggingFace model.

## Endpoints

- `GET /` - Status check
- `GET /health` - Health check
- `POST /chat` - Send question with PDF context

### POST /chat Request
```json
{
  "context": "Your PDF text content...",
  "question": "What is this document about?"
}
```

### Response
```json
{
  "success": true,
  "answer": "This document is about..."
}
```

## Local Development

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Server runs at http://localhost:8000
