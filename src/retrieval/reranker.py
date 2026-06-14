import sys
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    print("\n❌ Error: 'sentence-transformers' library missing. Run: pip install sentence-transformers")
    sys.exit(1)

# ❌ INCORRECT (Triggers the Hugging Face 401 / RepoNotFound Exception):
# class LocalReranker:
#     def __init__(self, model_name: str = "sentence-transformers/ms-marco-MiniLM-L-6-v2"):

#  CORRECT (Points directly to the official open cross-encoder repository):
class LocalReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """
        Initializes a localized Cross-Encoder reranking engine entirely offline.
        """
        print(f"🧠 Initializing Local Cross-Encoder Reranker: '{model_name}'...")
        self.model = CrossEncoder(model_name)
        print("✅ Reranking Model Loaded and Ready!")

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Calculates deep contextual relevance scores between the query and all candidates,
        re-sorts them, and returns the top_n absolute highest-quality results.
        """
        if not candidates:
            return []

        print(f"🔄 Reranking {len(candidates)} candidates down to Top-{top_n} via deep attention mapping...")

        # 1. Prepare pairs as required by Cross-Encoder signature layout: [[query, text1], [query, text2]]
        pairs = [[query, item["page_content"]] for item in candidates]

        # 2. Compute true attention-based relevance scores (values range generally between 0 and 1 or logit scales)
        scores = self.model.predict(pairs)

        # 3. Update candidates with their freshly calculated rerank scores
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        # 4. Re-sort candidates based on the new rerank_score in descending order
        # Also drop exact duplicate text content values on the fly to maximize token diversity
        seen_texts = set()
        unique_sorted_candidates = []
        
        sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        for item in sorted_candidates:
            text_hash = item["page_content"].strip()
            if text_hash not in seen_texts:
                seen_texts.add(text_hash)
                unique_sorted_candidates.append(item)

        print(f"🎯 Reranking complete. Dropped duplicate text windows down to {len(unique_sorted_candidates)} distinct matches.")
        return unique_sorted_candidates[:top_n]

if __name__ == "__main__":
    # Standalone sanity testing path
    test_query = "What are the common symptoms of a patient presenting with an acute clinical episode?"
    
    # Simulating duplicate candidate records returned from our Qdrant vector retrieval test
    mock_candidates = [
        {"id": 1, "score": 0.5730, "page_content": "Oral mucosa is still moist and well hydrated. Posterior pharynx is clear. LUNGS: Clear with good breath sounds.", "source_file": "mtsamples.csv"},
        {"id": 2, "score": 0.5730, "page_content": "Oral mucosa is still moist and well hydrated. Posterior pharynx is clear. LUNGS: Clear with good breath sounds.", "source_file": "mtsamples.csv"},
        {"id": 3, "score": 0.5730, "page_content": "Patient reports worsening acute shortness of breath, sudden chest pain, and mild dizziness over the past 4 hours.", "source_file": "mtsamples.csv"}
    ]
    
    reranker = LocalReranker()
    final_results = reranker.rerank(query=test_query, candidates=mock_candidates, top_n=3)
    
    print("\n📊 --- Final Hardened Rerank Results ---")
    for idx, match in enumerate(final_results):
        print(f"[{idx + 1}] [Rerank Score: {match['rerank_score']:.4f}] | Original Vector Score: {match['score']:.4f} | Source: {match['source_file']}")
        print(f"Text Content: {match['page_content']}\n")