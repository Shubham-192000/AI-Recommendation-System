"""
main.py

FastAPI service -- 2 endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → conversation history leke agent reply + recommendations

WHY STATELESS?
Har /chat call mein poori conversation history aa jaati hai (messages array).
Server apni taraf kuch store nahi karta. Iska fayda:
  - Scale easily karo (koi bhi instance koi bhi request handle kar sakta hai)
  - Test karna aasaan hai (ek request = ek isolated unit)
  - PDF ka requirement yahi hai
"""

import os
import sys

# Add app directory to path so imports work
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()  # .env file se GROQ_API_KEY load karo

from agent import run_agent
from retrieval import get_retriever

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent that recommends SHL assessments",
    version="1.0.0",
)

# CORS allow karo taaki agar koi frontend banao to block na ho
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ #
# Request / Response schemas                                           #
# ------------------------------------------------------------------ #

class Message(BaseModel):
    role: str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class RecommendationItem(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: List[RecommendationItem]
    end_of_conversation: bool


# ------------------------------------------------------------------ #
# Startup: load retriever once so first /chat isn't slow              #
# ------------------------------------------------------------------ #

@app.on_event("startup")
async def startup_event():
    print("[startup] Pre-loading catalog retriever...")
    get_retriever()  # builds FAISS index on startup, not on first request
    print("[startup] Ready.")


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@app.get("/health")
def health():
    """
    GET /health → {"status": "ok"}
    PDF requirement: must return HTTP 200 with this exact body.
    Evaluator hits this first, allows up to 2 min for cold start.
    """
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    POST /chat → agent reply + optional recommendations

    Request body:
      { "messages": [{"role": "user", "content": "..."}, ...] }

    Response body:
      {
        "reply": "...",
        "recommendations": [{"name": "...", "url": "...", "test_type": "..."}],
        "end_of_conversation": false
      }

    TURN CAP: PDF says evaluator caps at 8 turns (user + assistant combined).
    We enforce this: if messages >= 8, we force end_of_conversation.
    """

    # Validate turn count (PDF: max 8 turns total including user + assistant)
    if len(request.messages) >= 8:
        # Force a final recommendation with whatever we have
        pass  # agent will handle naturally; we'll add guard below after agent call

    # Convert Pydantic models to plain dicts for agent
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]

    # Basic validation: last message should be from user
    if not messages_dicts or messages_dicts[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="Last message must be from user")

    try:
        result = run_agent(messages_dicts)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"[chat] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Force end if we've hit the turn cap
    if len(request.messages) >= 7:
        result.end_of_conversation = True

    return ChatResponse(
        reply=result.reply,
        recommendations=[
            RecommendationItem(name=r.name, url=r.url, test_type=r.test_type)
            for r in result.recommendations
        ],
        end_of_conversation=result.end_of_conversation,
    )
