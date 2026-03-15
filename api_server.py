#!/usr/bin/env python3
"""Backend API for Chad Gutierrez portfolio chatbot.

Uses OpenRouter API (compatible with OpenAI SDK) for model flexibility.
Set OPENROUTER_API_KEY env var before running.
"""
import json
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel

# Load knowledge base
knowledge = json.loads(Path(__file__).with_name("knowledge.json").read_text())

SYSTEM_PROMPT = f"""You are Chad Gutierrez's professional portfolio assistant. You help recruiters, hiring managers, and other visitors learn about Chad's background, experience, and qualifications.

You have access to Chad's complete professional profile below. Answer questions accurately based on this information. Be conversational, helpful, and professional. Keep responses concise (2-4 sentences for simple questions, more for detailed ones).

If asked something not covered in the profile, say you don't have that information and suggest the visitor reach out to Chad directly.

IMPORTANT RULES:
- Never fabricate information not in the profile
- Be enthusiastic but honest about Chad's qualifications
- When discussing AI skills, be transparent that many AI projects are hands-on builds from personal initiative and self-directed learning
- Highlight the unique combination of coaching + building that makes Chad distinctive
- If asked about salary, say Chad is targeting $175K+ depending on the role and total compensation package
- When appropriate, mention Chad's $3.2M+ business impact, 80.7% throughput gains, and other concrete metrics

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage per visitor
conversations: dict[str, list] = {}

class ChatRequest(BaseModel):
    message: str
    visitor_id: str = "anonymous"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    vid = req.visitor_id
    if vid not in conversations:
        conversations[vid] = []

    conversations[vid].append({"role": "user", "content": req.message})

    # Keep last 20 messages to manage context
    history = conversations[vid][-20:]

    model = os.environ.get("CHAT_MODEL", DEFAULT_MODEL)

    async def generate():
        full_response = ""
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

        conversations[vid].append({"role": "assistant", "content": full_response})
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/api/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
