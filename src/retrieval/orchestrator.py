import sys
from typing import List, Dict, Any

# Import Stage 1 and Stage 2 retrieval components
from src.retrieval.search_engine import retrieve_candidates
from src.retrieval.reranker import LocalReranker

class ClinicalRAGOrchestrator:
    def __init__(self) -> None:
        """
        Master RAG Orchestrator that unifies Two-Stage Retrieval:
        Stage 1: Bi-Encoder Vector Search (Qdrant)
        Stage 2: Cross-Encoder Reranker (Hugging Face)
        """
        print("🚀 Initializing Production Clinical RAG Orchestrator Engine...")
        # Initialize our local cross-encoder engine cleanly
        self.reranker = LocalReranker()
        print("✅ Orchestrator Layer fully synchronized and online!")

    def query(self, user_question: str, fetch_k: int = 25, return_n: int = 5) -> List[Dict[str, Any]]:
        """
        Executes the hardened two-stage semantic lookup pipeline.
        """
        # 1. Execute Stage 1: Fast Vector Search across 22,326 points
        print(f"\n[STAGE 1] Fetching top {fetch_k} candidates from Qdrant vector volume...")
        candidates = retrieve_candidates(query_text=user_question, top_k=fetch_k)
        
        if not candidates:
            print("⚠️ Ingestion Warning: No candidates extracted during vector space traversal.")
            return []

        # 2. Execute Stage 2: Deep Context Attention Reranking & Deduplication
        print(f"\n[STAGE 2] Passing candidates through Cross-Encoder attention scoring...")
        final_context_blocks = self.reranker.rerank(
            query=user_question, 
            candidates=candidates, 
            top_n=return_n
        )
        
        return final_context_blocks

if __name__ == "__main__":
    # Initialize the master Two-Stage Engine once outside the execution loop
    pipeline = ClinicalRAGOrchestrator()
    
    print("\n💬 Clinical Knowledge Base Query Interface Active.")
    print("Type your clinical question and hit Enter. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            # Capture dynamic runtime console user input text
            user_input = input("🧑‍⚕️ Enter Question: ")
            
            # Check for user exit flag commands
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 Shutting down RAG interface session. Standby.")
                break
                
            if not user_input.strip():
                continue
                
            # Process query execution across vector search and cross-attention ranking
            context_results = pipeline.query(user_question=user_input, fetch_k=25, return_n=3)
            
            print("\n👑 ================= CONTEXT RETRIEVAL RESULTS =================")
            if not context_results:
                print("❌ No matching clinical contexts found inside Qdrant storage volume.")
            else:
                for idx, chunk in enumerate(context_results):
                    print(f"\n🎯 [Rank {idx + 1}] | Cross-Encoder Relevance Logit Score: {chunk['rerank_score']:.4f}")
                    print(f"📦 Source File: {chunk['source_file']} | Postgres Chunk ID: {chunk['id']}")
                    print(f"📄 Verified Context: {chunk['page_content']}")
                    print("-" * 70)
            print("\n" + "="*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Session interrupted via keyboard shortcut. Exiting.")
            break