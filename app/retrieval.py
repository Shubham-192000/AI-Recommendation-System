import json, re
from typing import List, Dict
from rank_bm25 import BM25Okapi

CATALOG_PATH = "data/catalog_clean.json"

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]

class CatalogRetriever:
    def __init__(self):
        print("[retrieval] Loading catalog...")
        with open(CATALOG_PATH) as f:
            self.catalog: List[Dict] = json.load(f)
        print(f"[retrieval] {len(self.catalog)} entries loaded.")
        tokenized = [tokenize(e["searchable_text"]) for e in self.catalog]
        self.bm25 = BM25Okapi(tokenized)
        print("[retrieval] BM25 index ready.")

    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        if not query.strip():
            return []
        scores = self.bm25.get_scores(tokenize(query))
        import numpy as np
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in sorted_idx[:top_k * 5]:
            entry = self.catalog[idx].copy()
            entry["_score"] = float(scores[idx])
            if filters:
                if filters.get("test_type"):
                    if not any(t in entry["test_type"] for t in filters["test_type"]):
                        continue
                if filters.get("remote_only") and not entry["remote_testing"]:
                    continue
            results.append(entry)
            if len(results) >= top_k:
                break
        return results

_retriever = None

def get_retriever() -> CatalogRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CatalogRetriever()
    return _retriever