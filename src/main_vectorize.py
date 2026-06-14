import sys
from typing import List, Dict, Any
from qdrant_client.models import PointStruct

from src.database.postgres import SessionLocal, ChunkModel, DocumentModel
from src.database.vector_db import init_vector_collection, qdrant_client, COLLECTION_NAME
from src.embeddings.local_client import LocalEmbeddingClient

def run_vectorization_pipeline(batch_size: int = 64):
    """
    Queries raw text chunks out of PostgreSQL relational tracking records,
    computes localized 384-dimensional vector weights completely offline,
    and uploads synchronized search payloads directly into Qdrant.
    """
    print("🚀 Initializing Vector Generation & Sync Pipeline...")
    
    # Ensure our collection is initialized in the Qdrant container
    init_vector_collection()
    
    # Initialize the local Hugging Face machine learning client
    embed_client = LocalEmbeddingClient()
    
    # Spin up an active structural persistence transaction socket
    db_session = SessionLocal()
    
    try:
        print("📥 Querying text chunks from PostgreSQL Relational Engine...")
        # Join chunks with documents to inherit parent metadata like filenames
        records = (
            db_session.query(ChunkModel)
            .join(DocumentModel, ChunkModel.document_id == DocumentModel.id)
            .all()
        )
        
        total_chunks = len(records)
        if total_chunks == 0:
            print("⚠️ Ingestion State Alert: Found 0 database rows inside your text chunks table. Run 'python -m src.main_ingest' first.")
            return

        print(f"📊 Detected {total_chunks} chunk blocks awaiting mathematical vectorization.")
        
        # Slicing tracking blocks into clean batch steps to optimize memory usage
        for i in range(0, total_chunks, batch_size):
            batch_records = records[i : i + batch_size]
            
            # Target the true schema attribute column: page_content
            batch_texts: List[str] = [str(rec.page_content) for rec in batch_records]
            
            # Generate local multi-dimensional arrays entirely offline
            print(f"🧠 Computing localized tensor calculations for batch indices [{i} to {min(i + batch_size, total_chunks)}]...")
            batch_vectors = embed_client.get_embeddings_batch(batch_texts)
            
            qdrant_points = []
            for record, vector in zip(batch_records, batch_vectors):
                
                # Dynamic property fallback to bypass the SQLAlchemy .metadata naming conflict
                row_metadata = record.__dict__.get("metadata") or {}
                
                # Construct payload metadata aligned with production schema names
                metadata_payload = {
                    "document_id": record.document_id,
                    "token_count": record.token_count,
                    "page_content": str(record.page_content),
                    "source_file": record.document.filename,
                    "metadata": row_metadata  # Safely injects the true primitive data dict
                }
                
                # Force runtime evaluation & declare a strict primitive integer for Pylance
                point_id: int = int(getattr(record, "id"))
                
                # Append structured Point layout map matching Qdrant protocol requirements
                qdrant_points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=metadata_payload
                    )
                )
            
            # Fire batch packet into the listening Qdrant container socket
            print(f"📤 Streaming {len(qdrant_points)} vector points into Qdrant collection...")
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=qdrant_points
            )
            
        print(f"\n🎉 Success! All {total_chunks} data points have been calculated and synchronized inside Qdrant.")

    except Exception as e:
        print(f"❌ Pipeline Execution Error encountered: {e}")
        raise e
    finally:
        # Guarantee transaction channels close to avoid pooling leaks
        db_session.close()

if __name__ == "__main__":
    run_vectorization_pipeline()