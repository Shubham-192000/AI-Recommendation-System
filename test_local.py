"""
test_local.py

Groq API key ke bina bhi retrieval test kar sakte ho.
With API key: full agent test.

Run: python3 test_local.py
"""
import sys
import os
import json

sys.path.insert(0, "app")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("TEST 1: Retrieval only (no API key needed)")
print("=" * 60)

from retrieval import get_retriever
retriever = get_retriever()

query = "personality test for senior manager selection"
results = retriever.search(query, top_k=5)
print(f"\nQuery: '{query}'")
print(f"Top {len(results)} results:")
for r in results:
    print(f"  - {r['name']} [{', '.join(r['test_type'])}] score={r['_score']:.3f}")

query2 = "java developer coding test"
results2 = retriever.search(query2, top_k=5)
print(f"\nQuery: '{query2}'")
print(f"Top {len(results2)} results:")
for r in results2:
    print(f"  - {r['name']} [{', '.join(r['test_type'])}] score={r['_score']:.3f}")

# Test with filter
results3 = retriever.search("leadership assessment", top_k=5, filters={"test_type": ["P"]})
print(f"\nQuery: 'leadership assessment' with filter test_type=P")
print(f"Top {len(results3)} results:")
for r in results3:
    print(f"  - {r['name']} [{', '.join(r['test_type'])}]")

print("\n" + "=" * 60)
print("TEST 2: Full agent test (needs GROQ_API_KEY in .env)")
print("=" * 60)

if not os.path.exists(".env"):
    print("Skipping -- no .env file found. Create .env with GROQ_API_KEY=xxx")
else:
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        print("Skipping -- GROQ_API_KEY not set in .env")
    else:
        from agent import run_agent

        # Simulate C1 conversation: vague start
        print("\nConversation test:")
        msgs = [{"role": "user", "content": "I need an assessment for a senior manager"}]
        r = run_agent(msgs)
        print(f"Turn 1 reply: {r.reply}")
        print(f"Recs (should be empty for vague query): {r.recommendations}")

        # Add context
        msgs.append({"role": "assistant", "content": r.reply})
        msgs.append({"role": "user", "content": "Selection purpose, Director level, English only"})
        r2 = run_agent(msgs)
        print(f"\nTurn 2 reply: {r2.reply}")
        print(f"Recommendations ({len(r2.recommendations)}):")
        for rec in r2.recommendations:
            print(f"  - {rec.name} ({rec.test_type}) → {rec.url}")

print("\nAll tests done.")
