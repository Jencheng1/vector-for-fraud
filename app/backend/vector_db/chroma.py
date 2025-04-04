import chromadb
import numpy as np
from typing import List, Dict, Any
from .base import VectorDBInterface

class ChromaDB(VectorDBInterface):
    def __init__(self):
        self.client = None
        self.collection = None
        self.dimension = 128  # Default dimension for vectors
        
    def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="fraud_collection",
            metadata={"dimension": self.dimension}
        )
        
    def insert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Insert vectors with their metadata into ChromaDB."""
        if self.collection is None:
            self.initialize()
            
        # Convert vectors to list format if they're numpy arrays
        vectors_list = [v.tolist() for v in vectors]
        
        # Prepare documents and ids
        documents = [str(i) for i in range(len(vectors))]
        ids = [f"vec_{i}" for i in range(len(vectors))]
        
        # Add vectors to collection
        self.collection.add(
            embeddings=vectors_list,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata."""
        if self.collection is None:
            return []
            
        # Convert query vector to list format
        query_vector = query_vector.tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k
        )
        
        # Return metadata for the found vectors
        return results['metadatas'][0]
        
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        if self.collection is None:
            return
            
        # Delete vectors from collection
        self.collection.delete(ids=ids)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.collection is None:
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "is_initialized": False
            }
            
        return {
            "total_vectors": self.collection.count(),
            "dimension": self.dimension,
            "is_initialized": True
        } 