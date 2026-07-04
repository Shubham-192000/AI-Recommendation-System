# SHL Assessment Recommender

A conversational agent that recommends SHL assessments based on hiring needs. Built with FastAPI, hybrid BM25 + TF-IDF retrieval, and a Groq-powered LLM for the conversation layer.

Instead of manually filtering SHL's product catalog, you describe what you're hiring for (e.g. "I need to hire a Java developer") and the agent asks follow-up questions and returns relevant assessments.

Live demo: https://web-production-4b351.up.railway.app/docs
Repo: github.com/Shubham-192000/AI-Recommendation-System

## Architecture

```
User message
     |
     v
FastAPI /chat endpoint
     |
     v
agent.py -> Groq LLM interprets intent, asks clarifying questions
     |
     v
retrieval.py -> BM25 + TF-IDF search over SHL catalog
     |
     v
Ranked recommendations + conversational reply
```

## Tech stack

- FastAPI - REST API layer
- Groq - LLM inference for conversation
- BM25 + TF-IDF - hybrid retrieval/ranking over catalog
- Railway - deployment
- Python 3

## Setup

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Configure environment
```bash
cp .env.example .env
```
Edit `.env` and add your Groq API key. Get one free at console.groq.com.

3. Process the catalog
```bash
python3 data/clean_catalog.py
```

4. Run locally
```bash
uvicorn app.main:app --reload --port 8000
```

5. Test retrieval (no API key needed)
```bash
python3 test_local.py
```

## API

**GET /health**
```json
{ "status": "ok" }
```

**POST /chat**

Request:
```json
{
  "messages": [
    { "role": "user", "content": "I need to hire a Java developer" }
  ]
}
```

Response:
```json
{
  "reply": "Sure! What seniority level are you targeting?",
  "recommendations": [],
  "end_of_conversation": false
}
```

## Deployment

Deployed on Railway with one environment variable:
```
GROQ_API_KEY=your_key
```

Steps:
1. Push repo to GitHub
2. Connect Railway (or Render) to the repo
3. Set `GROQ_API_KEY` env variable
4. Deploy

## Project structure

```
shl_recommender/
├── app/
│   ├── main.py                  # FastAPI endpoints (/health, /chat)
│   ├── agent.py                 # LLM prompt design + response parsing
│   └── retrieval.py              # BM25 + TF-IDF catalog search
├── data/
│   ├── clean_catalog.py          # One-time catalog cleaning script
│   ├── catalog_clean.json        # Cleaned catalog (generated)
│   └── shl_product_catalog.json  # Raw scraped data
├── test_local.py                 # Local retrieval test (no API key needed)
├── requirements.txt
├── Procfile                      # Railway/Render deployment config
└── .env.example
```

## Author

Shubham Yadav
AI Engineer @ TCS
linkedin.com/in/shubham-yadav-236745202 · github.com/Shubham-192000
