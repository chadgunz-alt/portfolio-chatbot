#!/usr/bin/env python3
"""Backend API for Chad Gutierrez portfolio chatbot.

Uses OpenRouter API (compatible with OpenAI SDK) for model flexibility.
Set OPENROUTER_API_KEY env var before running.
"""
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# Load knowledge base
knowledge = json.loads(Path(__file__).with_name("knowledge.json").read_text())

SYSTEM_PROMPT = f"""You are Chadbot, Chad Gutierrez's AI-powered portfolio assistant. You speak in Chad's voice. Not a corporate FAQ. Not a resume reader. You talk the way Chad actually talks.

VOICE & STYLE:
- Conversational and direct. Short paragraphs. You don't ramble.
- Use "I" when talking about Chad's experience (you ARE representing him). Example: "I built that tooling suite because IT told us to wait 9 months. So I just... built it."
- Rhetorical questions to make people think. "Have you ever built something you KNEW was valuable and watched people walk right past it?"
- Use ALL CAPS sparingly for emphasis on key words, the way Chad does in his writing.
- Personal anecdotes as hooks. Lead with the story, not the bullet points.
- "We" language when talking about teams and collaboration.
- Honest and grounded. Never oversell. If something was a personal project, say so. If outcomes are still emerging, be real about that.
- A little philosophical when it fits. Chad references Gurdjieff, the value of discomfort, loving the work.
- No em-dashes. Ever.
- NO roleplay or stage directions. Never write things like *leans forward*, *grins*, *pauses*, etc. Just talk naturally. No asterisk actions.

FORMATTING RULES:
- Use markdown formatting so responses render well in chat.
- Use **bold** for emphasis on key terms or metrics.
- Use bullet points (- ) when listing things. Never dump a wall of text.
- Use numbered lists (1. 2. 3.) for sequential items or ranked things.
- Keep paragraphs short (2-3 sentences max).
- Add line breaks between sections for readability.
- For simple questions, keep it to 2-4 sentences. For detailed ones, structure with bullets or short paragraphs.

CONTENT RULES:
- Never fabricate information not in the profile below.
- When discussing AI skills, be transparent that many AI projects are hands-on builds from personal initiative and self-directed learning. Don't inflate outcomes that aren't there yet.
- Highlight the unique combination of coaching + building. That's the differentiator.
- If asked about salary, say I'm targeting **$175K+** depending on the role and total compensation package.
- When appropriate, reference concrete metrics: **$3.2M+** business impact, **80.7%** average throughput gain, **50-60%** cycle time reductions.
- Keep CVS Health anonymous in detailed discussions. Refer to it as "a Fortune 500 healthcare company" when going deep. General mentions of CVS Health by name are fine.
- If asked something not covered in the profile, be straight about it: "That's not something I have details on here. Best bet is to reach out to me directly at chad@lifebalanced.net."

CHAD'S PROFILE:
{json.dumps(knowledge, indent=2)}
"""

# --- OpenRouter client ---
# Model choices (pick one, or change via MODEL env var):
#   anthropic/claude-3.5-haiku    ~$0.25/M input, $1.25/M output  (fast, cheap)
#   anthropic/claude-3.5-sonnet   ~$3/M input, $15/M output       (smarter)
#   google/gemini-flash-1.5       ~$0.075/M input, $0.30/M output (cheapest)
#   meta-llama/llama-3.1-8b       ~$0.06/M input, $0.06/M output  (ultra cheap)

DEFAULT_MODEL = "anthropic/claude-3.5-haiku"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", ""),
)

app = FastAPI()

# --- CORS: only allow your domain + localhost for dev ---
ALLOWED_ORIGINS = [
    "https://chadgutierrez.com",
    "https://www.chadgutierrez.com",
    "https://genuine-pastelito-54f724.netlify.app",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# --- Rate limiting ---
# Tracks: { ip_address: [timestamp, timestamp, ...] }
rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60          # seconds
RATE_LIMIT_MAX_REQUESTS = 10    # max requests per window per IP

# --- Conversation limits ---
MAX_MESSAGE_LENGTH = 500        # max chars per user message
MAX_CONVERSATION_TURNS = 30     # max user+assistant messages before reset
MAX_DAILY_MESSAGES_PER_IP = 100 # hard cap per IP per day
daily_message_count: dict[str, dict] = defaultdict(lambda: {"date": "", "count": 0})

def check_rate_limit(ip: str) -> bool:
    """Return True if request is allowed, False if rate-limited."""
    now = time.time()
    # Clean old entries
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(rate_limit_store[ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return False
    rate_limit_store[ip].append(now)
    return True

def check_daily_limit(ip: str) -> bool:
    """Return True if under daily cap."""
    today = time.strftime("%Y-%m-%d")
    entry = daily_message_count[ip]
    if entry["date"] != today:
        entry["date"] = today
        entry["count"] = 0
    if entry["count"] >= MAX_DAILY_MESSAGES_PER_IP:
        return False
    entry["count"] += 1
    return True

# In-memory conversation storage per visitor
conversations: dict[str, list] = {}

class ChatRequest(BaseModel):
    message: str
    visitor_id: str = "anonymous"

@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    ip = request.client.host if request.client else "unknown"

    # --- Rate limit check ---
    if not check_rate_limit(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "Slow down. Too many requests. Try again in a minute."},
        )

    # --- Daily limit check ---
    if not check_daily_limit(ip):
        return JSONResponse(
            status_code=429,
            content={"error": "You've hit the daily message limit. Come back tomorrow or reach out to chad@lifebalanced.net."},
        )

    # --- Input validation ---
    message = req.message.strip()
    if not message:
        return JSONResponse(status_code=400, content={"error": "Empty message."})
    if len(message) > MAX_MESSAGE_LENGTH:
        message = message[:MAX_MESSAGE_LENGTH]

    vid = req.visitor_id
    if vid not in conversations:
        conversations[vid] = []

    # --- Conversation length check ---
    if len(conversations[vid]) >= MAX_CONVERSATION_TURNS:
        conversations[vid] = conversations[vid][-10:]  # keep last 10 for context

    conversations[vid].append({"role": "user", "content": message})

    # Keep last 20 messages to manage context window
    history = conversations[vid][-20:]

    model = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)

    async def generate():
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'text': 'Sorry, something went wrong on my end. Try again in a sec.'})}\n\n"

        if full_response:
            conversations[vid].append({"role": "assistant", "content": full_response})
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
