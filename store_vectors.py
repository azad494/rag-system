#!/usr/bin/env python3
"""
Project 01: Production RAG System - Structural Chunking & Local Vector DB Store
Author: Lead AI Engineer & Project Manager
Description: Extracts text data, chunks it recursively, converts it to dense 
             384-D vectors, and commits it permanently to a local SQLite vector 
             database file. Safely resolves PyTorch/NumPy tensor casting issues.
"""

import sys
import sqlite3
from pathlib import Path

try:
    import numpy as np
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer
    import sqlite_vec
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e.name}")
    print("[INFO] Please run: pip install sentence-transformers langchain-text-splitters sqlite-vec numpy")
    sys.exit(1)

# Paths Configuration
RAW_DIR = Path("data/raw")
DB_PATH = Path("data/rag_storage.db")

def extract_text_from_pantry() -> list[tuple[str, str]]:
    """Reads txt and html files from pantry, returning pairs of (source_name, content)."""
    documents = []
    text_files = list(RAW_DIR.glob("*.txt")) + list(RAW_DIR.glob("*.html"))
    
    if not text_files:
        print("📭 No text documents found in 'data/raw/'. Run your automate_sourcing.py script first!")
        return []
        
    for file_path in text_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            if content.strip():
                documents.append((file_path.name, content))
        except Exception as e:
            print(f"⚠️ Warning: Skipping {file_path.name} due to read error: {e}")
            
    return documents

def init_vector_db():
    """Initializes local SQLite instance and registers the vector extension virtual tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(DB_PATH))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    
    cursor = conn.cursor()
    
    # Traditional relational metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT,
            text_content TEXT
        );
    """)
    
    # Virtual vector index table (384-D matches all-MiniLM-L6-v2)
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[384]
        );
    """)
    
    conn.commit()
    return conn

def run_pipeline():
    print("\n" + "="*60)
    print("      🧠 PIPELINE: RECURSIVE CHUNKING & VECTOR STORAGE 🧠")
    print("="*60)
    
    # 1. Gather files
    documents = extract_text_from_pantry()
    if not documents:
        return
        
    # 2. Text Chunking
    print("\n✂️  Applying Recursive Character Splitter (Size: 500, Overlap: 50)...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    all_chunks = []
    for source_file, text in documents:
        split_chunks = splitter.split_text(text)
        for chunk in split_chunks:
            all_chunks.append((source_file, chunk))
            
    print(f"✅ Generated {len(all_chunks)} contextual data chunks.")
    
    # 3. Embedding Vector Generation
    print("\n⏳ Loading 'all-MiniLM-L6-v2' vector embedding model locally...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    pure_texts = [item[1] for item in all_chunks]
    print(f"⚡ Computing multi-dimensional vector weights for {len(pure_texts)} chunks...")
    
    # 💡 FIX: Explicitly pass convert_to_numpy=True to prevent PyTorch Tensor outputs
    embeddings = model.encode(pure_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # 4. Stream and Commit to Local Database File
    print(f"\n🗄️  Connecting to local vector index store: '{DB_PATH.name}'...")
    conn = init_vector_db()
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM document_chunks;")
    cursor.execute("DELETE FROM vec_chunks;")
    
    for idx, (source_file, text_content) in enumerate(all_chunks):
        # Insert human-readable structure data
        cursor.execute(
            "INSERT INTO document_chunks (source_file, text_content) VALUES (?, ?);",
            (source_file, text_content)
        )
        generated_id = cursor.lastrowid
        
        # 💡 DEFENSIVE CASTING GUARD: Convert row dynamically into a 32-bit floating NumPy vector.
        # This completely clears any outstanding "tobytes is unknown" Pylance issues.
        vector_array = np.array(embeddings[idx], dtype=np.float32)
        vector_blob = vector_array.tobytes()
        
        # Insert vector row matching the same primary key mapping ID
        cursor.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?);",
            (generated_id, vector_blob)
        )
        
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("📊 VECTOR PIPELINE SUCCESSFULLY LOADED!")
    print(f" -> Stored Database Path: {DB_PATH}")
    print(f" -> Matrix Dimensions Committed: {embeddings.shape}")
    print("="*60)
    print("🎉 Data successfully indexed! Ready for Phase 4 (Semantic Search).")

if __name__ == "__main__":
    run_pipeline()