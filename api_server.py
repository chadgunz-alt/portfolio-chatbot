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
