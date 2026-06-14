import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# Initialize the SQLAlchemy Declarative Base Class
Base = declarative_base()

class DocumentModel(Base):
    """
    Parent Model representing the primary metadata tracking table 
    for raw ingested domain files.
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)  # Fixed typo here
    filename = Column(String(255), nullable=False, unique=True)
    file_type = Column(String(50), nullable=False)
    raw_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relational Child Mapping Link
    chunks = relationship("ChunkModel", back_populates="document", cascade="all, delete-orphan")


class ChunkModel(Base):
    """
    Child Model representing individual sliced text blocks,
    natively structured for high-performance context filtering.
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_content = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    
    # JSONB columns enable optimized GIN indexing for quick metadata key filtration
    metadata_fields = Column(JSONB, nullable=False, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

    # Inverse Relationship Mapping
    document = relationship("DocumentModel", back_populates="chunks")


# Database Connectivity Initialization Layer
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rag_metadata")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_postgres_tables():
    """Generates schema model tables within the targeted database instance context."""
    Base.metadata.create_all(bind=engine)
    print("📁 PostgreSQL Relational Storage Schema Successfully Initialized!")