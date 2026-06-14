import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingEngine:
    """
    Handles chunking of raw extracted text layers using token-aware
    recursive character separation mechanisms to preserve contextual integrity.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, model_name: str = "gpt-4"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Load the tiktoken tokenizer to cleanly monitor operational payload metrics
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            
        # Instantiate the recursive splitter utilizing character arrays
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self.get_token_count,
            separators=["\n\n", "\n", " ", ""]
        )

    def get_token_count(self, text: str) -> int:
        """Calculates precise length metrics for targeted validation."""
        return len(self.tokenizer.encode(text))

    def split_document(self, doc_payload: dict) -> list[dict]:
        """
        Takes an extracted document dictionary payload, slices it into structured,
        overlapping chunks, and calculates exact validation metadata.
        """
        raw_text = doc_payload.get("raw_text", "")
        if not raw_text.strip():
            return []

        # Generate text splits
        text_chunks = self.splitter.split_text(raw_text)
        
        processed_chunks = []
        for index, chunk_text in enumerate(text_chunks):
            token_len = self.get_token_count(chunk_text)
            
            # Map into a clean, relational tracking structure matching the DB layout
            processed_chunks.append({
                "chunk_index": index,
                "page_content": chunk_text,
                "token_count": token_len,
                "metadata_fields": {
                    "source_file": doc_payload.get("filename"),
                    "file_type": doc_payload.get("file_type"),
                    "processed_at": doc_payload.get("processed_at", "")
                }
            })
            
        # Log health check data per document split sequence
        print(f"✂️ Processed '{doc_payload.get('filename')}': Generated {len(processed_chunks)} text segments.")
        return processed_chunks