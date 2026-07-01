# SHL Assessment Recommender

Conversational agent that recommends SHL assessments via FastAPI.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file:
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
# Get free key from: https://console.groq.com
```

3. Process catalog:
```bash
python3 data/clean_catalog.py
```

4. Run locally:
```bash
uvicorn app.main:app --reload --port 8000
```

5. Test retrieval (no API key needed):
```bash
python3 test_local.py
```

## API

**GET /health** → `{"status": "ok"}`

**POST /chat**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"}
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

## Deployment (Railway/Render)

1. Push to GitHub
2. Connect Railway/Render to your repo
3. Set environment variable: `GROQ_API_KEY=your_key`
4. Deploy → your public URL is your submission URL

## Project Structure

```
shl_recommender/
├── app/
│   ├── main.py        # FastAPI endpoints (/health, /chat)
│   ├── agent.py       # LLM prompt design + response parsing
│   └── retrieval.py   # BM25+TF-IDF catalog search
├── data/
│   ├── clean_catalog.py         # One-time cleaning script
│   ├── catalog_clean.json       # Cleaned catalog (generated)
│   └── shl_product_catalog.json # Raw scraped data
├── requirements.txt
├── Procfile            # For Railway/Render deployment
├── .env.example
└── test_local.py       # Local testing
```
