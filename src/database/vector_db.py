import os
from typing import Any, List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "patient_clinical_notes"
VECTOR_DIMENSION = 384  # Configured precisely for local sentence-transformers all-MiniLM-L6-v2

qdrant_client: QdrantClient = QdrantClient(host="localhost", port=6333)

def init_vector_collection(force_recreate: bool = False) -> None:
    """
    Initializes the target Qdrant collection for dense vectors.
    Bypasses dimensional stale tracking by allowing an intentional recreation clear path.
    """
    print(f"📦 Verifying Qdrant Vector Collection: '{COLLECTION_NAME}'...")
    try:
        collections = qdrant_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if exists and force_recreate:
            print(f"⚠️ Stale context schema alignment found. Deleting collection '{COLLECTION_NAME}'...")
            qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
            exists = False
            
        if not exists:
            print(f"🛠️ Re-creating collection '{COLLECTION_NAME}' to accept 384 dimensional values...")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Successfully created clean Qdrant collection: '{COLLECTION_NAME}' (384 Dimensions).")
        else:
            print(f"⏩ Qdrant collection '{COLLECTION_NAME}' already exists. Skipping initialization.")
    except Exception as e:
        print(f"❌ Error during Qdrant collection initialization: {str(e)}")

def search(query_embedding: List[float], limit: int = 5) -> List[Any]:
    """
    Executes a high-performance semantic vector similarity search.
    Casts search outputs explicitly to satisfy the Pylance index checker.
    """
    try:
        client_any: Any = qdrant_client
        search_results: List[Any] = list(
            client_any.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_embedding,
                limit=limit
            )
        )
        return search_results
    except Exception as e:
        print(f"❌ Error executing vector search: {str(e)}")
        return []

if __name__ == "__main__":
    # Standalone script call defaults to forcing an instant environment configuration purge
    init_vector_collection(force_recreate=True)