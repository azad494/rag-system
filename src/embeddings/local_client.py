import os
# FIX: Made the import path explicit to completely satisfy Pylance
from sentence_transformers.sentence_transformer import SentenceTransformer

class LocalEmbeddingClient:
    """
    Adapter class to interface natively with an open-source, local embedding model.
    Utilizes 'all-MiniLM-L6-v2' to produce high-performance 384-dimensional
    vector coordinates completely offline, safeguarding clinical data privacy.
    """
    def __init__(self):
        # We explicitly lock onto the exact standard Hugging Face path
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        print(f"🧠 Initializing Local Embedding Engine: '{self.model_name}'...")
        
        # This will download the model weights automatically on the very first run
        # and cache them locally for all subsequent lightning-fast executions.
        self.client = SentenceTransformer(self.model_name)
        print("✅ Local Embedding Model Loaded and Ready!")

    def get_embedding(self, text: str) -> list[float]:
        """
        Converts a single block of plain text into a dense 384-dimensional vector list.
        """
        if not text.strip():
            raise ValueError("Cannot calculate mathematical vector values for empty text inputs.")
            
        # encode returns a numpy array; we cast it cleanly back to a native Python list of floats
        vector = self.client.encode(text)
        return vector.tolist()

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Batches multiple text segments to optimize local hardware processor utilization.
        """
        # Defensive guard: Ensure no completely empty string blocks cause a tensor calculation glitch
        clean_texts = [t if t.strip() else " " for t in texts]
        
        vectors = self.client.encode(clean_texts, show_progress_bar=False)
        return vectors.tolist()

if __name__ == "__main__":
    # Smoke-test to verify the local machine learning pipeline functions smoothly
    client = LocalEmbeddingClient()
    test_vector = client.get_embedding("Verify clinical RAG persistence layer connection.")
    print(f"🚀 Success! Generated vector array length: {len(test_vector)} dimensions.")