import faiss
import numpy as np
from typing import List, Dict, Any
from .base import VectorDBInterface

class FAISSDB(VectorDBInterface):
    def __init__(self):
        self.index = None
        self.metadata = []
        self.dimension = 128  # Default dimension for vectors
        
    def initialize(self) -> None:
        """Initialize FAISS index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        
    def insert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Insert vectors with their metadata into FAISS."""
        if self.index is None:
            self.initialize()
            
        # Convert vectors to numpy array if they aren't already
        vectors_array = np.array(vectors).astype('float32')
        
        # Add vectors to FAISS index
        self.index.add(vectors_array)
        
        # Store metadata
        self.metadata.extend(metadata)
        
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata."""
        if self.index is None:
            return []
            
        # Ensure query vector is in the correct format
        query_vector = query_vector.astype('float32').reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.index.search(query_vector, k)
        
        # Return metadata for the found vectors
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata):
                results.append(self.metadata[idx])
                
        return results
        
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        # FAISS doesn't support deletion, so we'll need to rebuild the index
        if not self.metadata:
            return
            
        # Keep only metadata that's not in the delete list
        new_metadata = []
        new_vectors = []
        
        for i, meta in enumerate(self.metadata):
            if meta['id'] not in ids:
                new_metadata.append(meta)
                # Get the vector from the original index
                vector = self.index.reconstruct(i)
                new_vectors.append(vector)
                
        # Rebuild the index with remaining vectors
        if new_vectors:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(np.array(new_vectors).astype('float32'))
        else:
            self.index = None
            
        self.metadata = new_metadata
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_vectors": len(self.metadata),
            "dimension": self.dimension,
            "is_initialized": self.index is not None
        } 