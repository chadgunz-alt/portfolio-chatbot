# Chad Gutierrez Portfolio Chatbot — Backend

A lightweight FastAPI server that powers the AI chatbot on your portfolio site. Uses **OpenRouter** so you can pick any model (Claude, Gemini, Llama, etc.) with one API key.

---

## What's in here

| File | Purpose |
|------|---------|
| `api_server.py` | FastAPI app — chat endpoint with SSE streaming |
| `knowledge.json` | Your professional profile (the chatbot's brain) |
| `requirements.txt` | Python dependencies |
| `Procfile` | Start command for Railway/Render |

---

## Quick Start (Local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."

# 3. Run the server
python api_server.py
```

Server starts at `http://localhost:8000`. Test it:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is Chad's experience?", "visitor_id": "test"}'
```

---

## Deploy to Railway (Recommended)

1. **Get an OpenRouter API key** at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Push this folder to a **GitHub repo**
3. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub Repo
4. Add environment variable: `OPENROUTER_API_KEY` = your key
5. Railway gives you a public URL like `https://your-app.up.railway.app`
6. Update your portfolio site's `app.js` — change the API URL on line 150:

```javascript
const API = "https://your-app.up.railway.app";
```

7. Redeploy the static site

---

## Deploy to Render (Alternative)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New Web Service → connect repo
3. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
4. Add env var: `OPENROUTER_API_KEY`
5. Update your portfolio's API URL (same as step 6 above)

---

## Choosing a Model

Default: `anthropic/claude-3.5-haiku` (fast, cheap, smart enough for Q&A)

Override via environment variable:

```bash
export CHAT_MODEL="google/gemini-flash-1.5"   # cheapest option
export CHAT_MODEL="anthropic/claude-3.5-sonnet" # smartest option
export CHAT_MODEL="meta-llama/llama-3.1-8b"    # open source, ultra cheap
```

### Estimated Monthly Cost (at ~50 recruiter conversations/month)

| Model | Approx. Cost |
|-------|-------------|
| Gemini Flash 1.5 | < $0.50/mo |
| Llama 3.1 8B | < $0.50/mo |
| Claude 3.5 Haiku | ~$1-2/mo |
| Claude 3.5 Sonnet | ~$5-10/mo |

Plus hosting: Railway free tier or ~$5/mo for always-on.

---

## Updating Your Profile

Edit `knowledge.json` and redeploy. The chatbot's responses will immediately reflect the changes. No code changes needed.

---

## Connecting to Your Portfolio Site

Your deployed portfolio site has a chatbot widget built in. It just needs to know where the backend lives. In `app.js` line 150, the API URL is set. Once you deploy this backend and get a public URL, update that line and redeploy the static site.
