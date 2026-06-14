import sys
from typing import List, Dict, Any

# Import our unified offline embedding client
from src.embeddings.local_client import LocalEmbeddingClient
# Import our verified search wrapper and vector configuration settings
from src.database.vector_db import search, COLLECTION_NAME

def retrieve_candidates(query_text: str, top_k: int = 25) -> List[Dict[str, Any]]:
    """
    Stage 1 Retrieval (Bi-Encoder Vector Search):
    Transforms a raw text query string into a local vector representation
    and fetches a broad candidate pool from the Qdrant database.
    """
    if not query_text.strip():
        print("⚠️ Warning: Received an empty query string.")
        return []

    print(f"\n🔍 Processing incoming query: '{query_text}'")
    
    try:
        # 1. Initialize local Hugging Face transformer model weights completely offline
        embed_client = LocalEmbeddingClient()
        
        # 2. Compute the query vector embedding coordinates (384 dimensions)
        print("🧠 Computing query tensor array...")
        # Wrap query text in a list since get_embeddings_batch expects an iterable sequence
        query_embeddings = embed_client.get_embeddings_batch([query_text])
        query_vector = query_embeddings[0]

        # 3. Query our active Qdrant Docker container using Cosine Similarity
        print(f"📡 Querying Qdrant collection '{COLLECTION_NAME}' for Top-{top_k} matches...")
        raw_hits = search(query_embedding=query_vector, limit=top_k)
        
        # 4. Parse the scored vector point results into a clean, structured payload map
        parsed_results: List[Dict[str, Any]] = []
        for hit in raw_hits:
            parsed_results.append({
                "id": hit.id,
                "score": hit.score,
                "page_content": hit.payload.get("page_content", ""),
                "source_file": hit.payload.get("source_file", "Unknown"),
                "document_id": hit.payload.get("document_id", None),
                "token_count": hit.payload.get("token_count", 0),
                "metadata": hit.payload.get("metadata", {})
            })
            
        print(f"✅ Stage 1 Complete: Successfully extracted {len(parsed_results)} candidate chunks.")
        return parsed_results

    except Exception as e:
        print(f"❌ Critical failure during vector candidate retrieval: {e}")
        return []

if __name__ == "__main__":
    # Standalone sanity test script block
    test_query = "What are the common symptoms of a patient presenting with an acute clinical episode?"
    results = retrieve_candidates(query_text=test_query, top_k=3)
    
    print("\n📊 --- Standing Query Hit Results ---")
    for idx, match in enumerate(results):
        print(f"\n[{idx + 1}] [Score: {match['score']:.4f}] | Source: {match['source_file']}")
        print(f"Content Snippet: {match['page_content'][:150]}...")