import json, os, re
from typing import List, Dict
from groq import Groq
from pydantic import BaseModel
from retrieval import get_retriever

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class AgentResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool

SYSTEM_PROMPT = """You are an SHL Assessment Recommender Agent. Help hiring managers find the right SHL assessments.

STRICT RULES:
1. ONLY discuss SHL assessments. Refuse general HR/legal/compliance questions politely.
2. NEVER recommend any test not in CATALOG CONTEXT. Every URL must come from catalog only.
3. Max 10 recommendations. Min 1 when you commit to a shortlist.
4. Vague query = ask ONE focused clarifying question. Never recommend on turn 1 if role/purpose unclear.
5. Refine: when user says "add X", "drop Y", "replace Z" — update shortlist precisely, keep rest unchanged.
6. Compare: explain differences using catalog data only. Maintain current recommendations alongside comparison.
7. end_of_conversation = TRUE only when user explicitly confirms: "Perfect", "Confirmed", "Locking it in", "That covers it", "That's good", "Confirmed", "Good", "Keep it". Set FALSE otherwise.
8. Legal/compliance questions (e.g. "Are we legally required...") → refuse and redirect to legal team.
9. If catalog has no exact match (e.g. no Rust test), say so honestly and suggest closest alternatives.

BEHAVIORS:
- CLARIFY: Ask focused questions — role type, seniority level, selection vs development, language needs, backend vs frontend etc.
- RECOMMEND: Once enough context, give grounded shortlist from catalog context only.
- REFINE: "Add AWS, drop REST" → update precisely. Do not restart. Keep unchanged items.
- COMPARE: Draw from catalog descriptions only. Keep showing current recommendations.
- REFUSE: Legal, salary, general HR advice → politely decline, stay in scope.

OUTPUT: Respond ONLY with valid JSON, no markdown:
{
  "reply": "<response>",
  "recommendations": [{"name": "<exact name>", "url": "<exact url>", "test_type": "<single letter>"}],
  "end_of_conversation": false
}
recommendations = [] when clarifying, comparing without new shortlist, or refusing."""

SATISFACTION_WORDS = ["thank", "thanks", "perfect", "great", "looks good", "that's all", "that is all", "ok done", "noted", "awesome", "done"]

def build_catalog_context(entries):
    if not entries:
        return "No catalog entries found."
    lines = ["=== CATALOG CONTEXT (only recommend from these) ===\n"]
    for i, e in enumerate(entries, 1):
        codes = ", ".join(e.get("test_type", []))
        levels = ", ".join(e.get("job_levels", [])) or "All levels"
        desc = e["description"][:250]
        lines.append(f"{i}. NAME: {e['name']}\n   URL: {e['url']}\n   TEST_TYPE_CODES: {codes}\n   JOB_LEVELS: {levels}\n   DESCRIPTION: {desc}...\n")
    return "\n".join(lines)

def extract_query(messages):
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    return " ".join(user_msgs[-3:])

def parse_response(raw_text):
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_text).strip().rstrip("`").strip()
    parsed = json.loads(cleaned)
    recs = []
    for r in parsed.get("recommendations", [])[:10]:
        tt = r.get("test_type", "")
        if isinstance(tt, list):
            tt = tt[0] if tt else ""
        recs.append({"name": r.get("name", ""), "url": r.get("url", ""), "test_type": tt})
    parsed["recommendations"] = recs
    return AgentResponse(**parsed)

def run_agent(messages: List[Dict]) -> AgentResponse:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")

    client = Groq(api_key=api_key)
    retriever = get_retriever()
    query = extract_query(messages)
    retrieved = retriever.search(query, top_k=10)
    catalog_context = build_catalog_context(retrieved)
    full_system = SYSTEM_PROMPT + "\n\n" + catalog_context

    messages_for_llm = [{"role": "system", "content": full_system}]
    for m in messages:
        messages_for_llm.append({"role": m["role"], "content": m["content"]})

    for attempt in range(2):
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages_for_llm,
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            raw_text = completion.choices[0].message.content
            result = parse_response(raw_text)
            last_msg = messages[-1]["content"].lower()
            if any(w in last_msg for w in SATISFACTION_WORDS) and result.recommendations:
                result.end_of_conversation = True
            return result
        except Exception as e:
            print(f"[agent] Attempt {attempt+1} failed: {e}")

    return AgentResponse(
        reply="Could you please rephrase your requirement?",
        recommendations=[],
        end_of_conversation=False,
    )