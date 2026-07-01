"""
retrieval.py

WHY TF-IDF + BM25 INSTEAD OF EMBEDDINGS?
Embeddings (sentence-transformers) need HuggingFace download. For production
deployment on Railway/Render we CAN use them. But TF-IDF+BM25 is actually
excellent for a structured catalog like SHL's because:
  - Exact test names matter ("OPQ32r", "Java 8 (New)") -- TF-IDF handles these perfectly
  - 377 entries is tiny -- TF-IDF is instant, no GPU needed
  - BM25 is the industry standard for catalog/document search (used by Elasticsearch)
  - No API cost, no download, works offline

HOW BM25 WORKS (simple explanation):
TF-IDF mein problem hai ki common words (the, is, for) high weight lete hain.
BM25 isko fix karta hai:
  - Rare words ko zyada weight (good)
  - Document length normalize karta hai (long doc pe zyada matches unfair na ho)
  - Diminishing returns: "java java java" != 3x zyada relevant than "java" once
"""

import json
import re
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CATALOG_PATH = "data/catalog_clean.json"


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: lowercase, split on non-alphanumeric, remove short tokens."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if len(t) > 1]


class CatalogRetriever:
    def __init__(self):
        print("[retrieval] Loading catalog...")
        with open(CATALOG_PATH) as f:
            self.catalog: List[Dict] = json.load(f)
        print(f"[retrieval] {len(self.catalog)} entries loaded.")

        # Build tokenized corpus for BM25
        self.corpus_texts = [e["searchable_text"] for e in self.catalog]
        tokenized_corpus = [tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Also build TF-IDF for hybrid scoring
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=True, token_pattern=None)
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus_texts)
        print("[retrieval] BM25 + TF-IDF index ready.")

    def search(self, query: str, top_k: int = 10, filters: Dict = None) -> List[Dict]:
        """
        Hybrid BM25 + TF-IDF search.
        
        BM25 aur TF-IDF dono scores nikaalte hain, normalize karte hain,
        combine karte hain (0.6 BM25 + 0.4 TF-IDF), fir filters apply karte hain.
        
        WHY HYBRID?
        BM25: exact keyword matching mein best
        TF-IDF: slightly different vocabulary handling
        Combined: more robust than either alone
        """
        if not query.strip():
            return []

        # BM25 scores
        bm25_scores = np.array(self.bm25.get_scores(tokenize(query)))

        # TF-IDF cosine similarity scores
        try:
            query_vec = self.tfidf.transform([query])
            tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        except Exception:
            tfidf_scores = np.zeros(len(self.catalog))

        # Normalize both to [0, 1]
        def normalize(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-9:
                return arr * 0
            return (arr - mn) / (mx - mn)

        combined = 0.6 * normalize(bm25_scores) + 0.4 * normalize(tfidf_scores)

        # Get sorted indices
        sorted_idx = np.argsort(combined)[::-1]

        results = []
        # Wider net for filtering
        fetch_k = min(top_k * 5, len(self.catalog))
        
        for idx in sorted_idx[:fetch_k]:
            entry = self.catalog[int(idx)].copy()
            entry["_score"] = float(combined[idx])

            # Apply filters
            if filters:
                if filters.get("test_type"):
                    if not any(t in entry["test_type"] for t in filters["test_type"]):
                        continue
                if filters.get("job_level"):
                    jl = filters["job_level"]
                    if entry["job_levels"] and jl not in entry["job_levels"]:
                        continue
                if filters.get("remote_only"):
                    if not entry["remote_testing"]:
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
