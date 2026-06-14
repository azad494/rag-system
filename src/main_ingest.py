import os
from datetime import datetime
from src.database.postgres import init_postgres_tables, SessionLocal, DocumentModel, ChunkModel
from src.ingestion.loaders import DocumentLoader
from src.ingestion.text_splitter import ChunkingEngine

def run_pipeline(raw_data_dir: str = "data/raw"):
    """
    Orchestrates the ETL execution pipeline: Extracts files from disk,
    Transforms text via token-aware chunk splitters, and Loads records
    into relational tables.
    """
    print("🚀 Initializing Master Data Ingestion Pipeline...")
    
    # Ensure database schemas are fully created before writing
    init_postgres_tables()
    
    # Check if raw data directory exists
    if not os.path.exists(raw_data_dir):
        print(f"❌ Error: The raw data directory '{raw_data_dir}' does not exist.")
        return

    # Instantiate our framework components
    loader = DocumentLoader()
    splitter = ChunkingEngine(chunk_size=512, chunk_overlap=64)
    db_session = SessionLocal()

    try:
        # 1. EXTRACT: Read files dynamically out of the raw directory
        print(f"📂 Scanning directory: {raw_data_dir}...")
        extracted_docs = loader.load_directory(raw_data_dir)
        
        if not extracted_docs:
            print("⚠️ No valid documents discovered inside the raw folder path.")
            return

        print(f"📄 Found {len(extracted_docs)} documents to process. Initializing DB persistence...")

        # 2. TRANSFORM & LOAD: Loop through and stream directly to PostgreSQL
        for doc_payload in extracted_docs:
            # Check if document already exists to avoid redundant parsing issues
            existing_doc = db_session.query(DocumentModel).filter_by(filename=doc_payload["filename"]).first()
            if existing_doc:
                print(f"⏩ Skipping '{doc_payload['filename']}' - Document already initialized in database.")
                continue

            # Map extracted content to our Parent Document Schema
            db_doc = DocumentModel(
                filename=doc_payload["filename"],
                file_type=doc_payload["file_type"],
                raw_text=doc_payload["raw_text"],
                created_at=datetime.utcnow()
            )
            
            # We add and flush here to generate the parent document ID for our child chunks
            db_session.add(db_doc)
            db_session.flush()

            # Append metadata timestamps for our chunk schema calculations
            doc_payload["processed_at"] = db_doc.created_at.isoformat()

            # 3. TRANSFORM: Slice raw text into optimized segments
            processed_chunks = splitter.split_document(doc_payload)

            # Map segments directly to our Child Chunks Schema
            for chunk in processed_chunks:
                db_chunk = ChunkModel(
                    document_id=db_doc.id,
                    chunk_index=chunk["chunk_index"],
                    page_content=chunk["page_content"],
                    token_count=chunk["token_count"],
                    metadata_fields=chunk["metadata_fields"],
                    created_at=db_doc.created_at
                )
                db_session.add(db_chunk)

        # Commit everything safely to the database instance
        db_session.commit()
        print("\n🎉 Pipeline Execution Completed Successfully! All records persisted to PostgreSQL.")

    except Exception as e:
        db_session.rollback()
        print(f"💥 Critical Error occurred during pipeline execution. Rolling back transactions: {e}")
        raise e
    finally:
        db_session.close()

if __name__ == "__main__":
    # Point execution context straight to your project's raw folder configuration
    run_pipeline()