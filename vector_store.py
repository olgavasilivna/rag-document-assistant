from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import uuid
import shutil

# Disable ChromaDB telemetry
os.environ["CHROMA_TELEMETRY"] = "false"

class VectorStore:
    def __init__(self, persist_directory: str = "chroma_db"):
        self.persist_directory = persist_directory
        
        # Initialize the client with in-memory settings
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            chroma_db_impl="duckdb",
            persist_directory="chroma_persist"
        ))
        
        # Initialize the collection
        self._initialize_collection()
        
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _initialize_collection(self):
        """Initialize or get the collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            # If there's an error, try to recreate the collection directly
            try:
                self.client.delete_collection("pdf_documents")
            except:
                pass
            self.collection = self.client.create_collection(
                name="pdf_documents",
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(self, documents: List[Dict]):
        """Add documents to the vector store."""
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, k: int = 3) -> List[Dict]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results["documents"][0])):
            formatted_results.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })
        
        return formatted_results
    
    def reset(self):
        """Reset the vector store."""
        # Reinitialize the client with in-memory settings
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            chroma_db_impl="duckdb",
            persist_directory="chroma_persist"
        ))
        
        # Create a new collection
        self._initialize_collection() 