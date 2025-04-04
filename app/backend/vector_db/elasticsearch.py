from elasticsearch import Elasticsearch
import numpy as np
from typing import List, Dict, Any
from .base import VectorDBInterface
import os
from dotenv import load_dotenv

class ElasticsearchDB(VectorDBInterface):
    def __init__(self):
        load_dotenv()
        self.client = None
        self.dimension = 128  # Default dimension for vectors
        self.url = os.getenv("ELASTICSEARCH_URL")
        self.username = os.getenv("ELASTICSEARCH_USERNAME")
        self.password = os.getenv("ELASTICSEARCH_PASSWORD")
        self.index_name = "fraud_transactions"
        
    def initialize(self) -> None:
        """Initialize Elasticsearch client and index."""
        if not self.url:
            raise ValueError("Elasticsearch URL must be set in .env file")
            
        # Create Elasticsearch client
        self.client = Elasticsearch(
            self.url,
            basic_auth=(self.username, self.password) if self.username and self.password else None
        )
        
        # Create index with vector mapping if it doesn't exist
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": self.dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "id": {"type": "keyword"},
                        "amount": {"type": "float"},
                        "timestamp": {"type": "date"},
                        "merchant": {"type": "keyword"},
                        "location": {"type": "keyword"},
                        "is_fraud": {"type": "boolean"},
                        "features": {"type": "object"}
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=mapping)
            
    def insert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Insert vectors with their metadata into Elasticsearch."""
        if self.client is None:
            self.initialize()
            
        # Convert vectors to list format if they're numpy arrays
        vectors_list = [v.tolist() for v in vectors]
        
        # Prepare documents for bulk insert
        bulk_data = []
        for i, (vector, meta) in enumerate(zip(vectors_list, metadata)):
            bulk_data.extend([
                {"index": {"_index": self.index_name, "_id": f"vec_{i}"}},
                {
                    "vector": vector,
                    **meta
                }
            ])
            
        # Insert documents in bulk
        if bulk_data:
            self.client.bulk(operations=bulk_data)
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata."""
        if self.client is None:
            return []
            
        # Convert query vector to list format
        query_vector = query_vector.tolist()
        
        # Search in Elasticsearch
        query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
        
        results = self.client.search(
            index=self.index_name,
            body={
                "size": k,
                "query": query,
                "_source": {
                    "excludes": ["vector"]
                }
            }
        )
        
        # Return metadata for the found vectors
        return [hit["_source"] for hit in results["hits"]["hits"]]
        
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        if self.client is None:
            return
            
        # Delete documents from Elasticsearch
        for doc_id in ids:
            self.client.delete(index=self.index_name, id=doc_id)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.client is None:
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "is_initialized": False
            }
            
        stats = self.client.indices.stats(index=self.index_name)
        count = stats["_all"]["total"]["docs"]["count"]
        
        return {
            "total_vectors": count,
            "dimension": self.dimension,
            "is_initialized": True
        } 