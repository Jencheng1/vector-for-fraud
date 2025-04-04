from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class VectorDBInterface(ABC):
    """Base interface for vector database implementations."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the vector database connection."""
        pass
    
    @abstractmethod
    def insert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Insert vectors with their metadata into the database."""
        pass
    
    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata."""
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass 