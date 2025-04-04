import pinecone
import numpy as np
from typing import List, Dict, Any
from .base import VectorDBInterface
import os
from dotenv import load_dotenv

class PineconeDB(VectorDBInterface):
    def __init__(self):
        load_dotenv()
        self.index = None
        self.dimension = 128  # Default dimension for vectors
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "fraud-index")
        
    def initialize(self) -> None:
        """Initialize Pinecone client and index."""
        if not self.api_key or not self.environment:
            raise ValueError("Pinecone API key and environment must be set in .env file")
            
        pinecone.init(api_key=self.api_key, environment=self.environment)
        
        # Create index if it doesn't exist
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine"
            )
            
        self.index = pinecone.Index(self.index_name)
        
    def insert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Insert vectors with their metadata into Pinecone."""
        if self.index is None:
            self.initialize()
            
        # Convert vectors to list format if they're numpy arrays
        vectors_list = [v.tolist() for v in vectors]
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (vector, meta) in enumerate(zip(vectors_list, metadata)):
            vectors_to_upsert.append({
                "id": f"vec_{i}",
                "values": vector,
                "metadata": meta
            })
            
        # Upsert vectors in batches of 100
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata."""
        if self.index is None:
            return []
            
        # Convert query vector to list format
        query_vector = query_vector.tolist()
        
        # Search in index
        results = self.index.query(
            vector=query_vector,
            top_k=k,
            include_metadata=True
        )
        
        # Return metadata for the found vectors
        return [match.metadata for match in results.matches]
        
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        if self.index is None:
            return
            
        # Delete vectors from index
        self.index.delete(ids=ids)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.index is None:
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "is_initialized": False
            }
            
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": self.dimension,
            "is_initialized": True
        } 