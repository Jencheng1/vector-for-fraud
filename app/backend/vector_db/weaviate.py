import weaviate
import numpy as np
from typing import List, Dict, Any
from .base import VectorDBInterface
import os
from dotenv import load_dotenv

class WeaviateDB(VectorDBInterface):
    def __init__(self):
        load_dotenv()
        self.client = None
        self.dimension = 128  # Default dimension for vectors
        self.url = os.getenv("WEAVIATE_URL")
        self.api_key = os.getenv("WEAVIATE_API_KEY")
        self.class_name = "FraudTransaction"
        
    def initialize(self) -> None:
        """Initialize Weaviate client and schema."""
        if not self.url:
            raise ValueError("Weaviate URL must be set in .env file")
            
        auth_config = weaviate.AuthApiKey(api_key=self.api_key) if self.api_key else None
        self.client = weaviate.Client(
            url=self.url,
            auth_client_secret=auth_config,
            additional_headers={
                "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY", "")
            }
        )
        
        # Create schema if it doesn't exist
        schema = {
            "class": self.class_name,
            "vectorizer": "none",  # We'll provide vectors ourselves
            "properties": [
                {"name": "id", "dataType": ["string"]},
                {"name": "amount", "dataType": ["number"]},
                {"name": "timestamp", "dataType": ["date"]},
                {"name": "merchant", "dataType": ["string"]},
                {"name": "location", "dataType": ["string"]},
                {"name": "is_fraud", "dataType": ["boolean"]},
                {"name": "features", "dataType": ["object"]}
            ]
        }
        
        try:
            self.client.schema.create_class(schema)
        except weaviate.exceptions.UnexpectedStatusCodeException:
            # Schema already exists
            pass
            
    def insert_vectors(self, vectors: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Insert vectors with their metadata into Weaviate."""
        if self.client is None:
            self.initialize()
            
        # Convert vectors to list format if they're numpy arrays
        vectors_list = [v.tolist() for v in vectors]
        
        # Prepare data objects
        data_objects = []
        for i, (vector, meta) in enumerate(zip(vectors_list, metadata)):
            data_objects.append({
                "class": self.class_name,
                "id": f"vec_{i}",
                "vector": vector,
                "properties": meta
            })
            
        # Insert objects in batches
        batch_size = 100
        for i in range(0, len(data_objects), batch_size):
            batch = data_objects[i:i + batch_size]
            self.client.batch.data_objects(batch)
            
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors and return their metadata."""
        if self.client is None:
            return []
            
        # Convert query vector to list format
        query_vector = query_vector.tolist()
        
        # Search in Weaviate
        result = (
            self.client.query
            .get(self.class_name)
            .with_near_vector({
                "vector": query_vector
            })
            .with_limit(k)
            .with_additional(["distance"])
            .do()
        )
        
        # Return metadata for the found vectors
        if "data" in result and "Get" in result["data"]:
            return [item["properties"] for item in result["data"]["Get"][self.class_name]]
        return []
        
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs."""
        if self.client is None:
            return
            
        # Delete objects from Weaviate
        for obj_id in ids:
            self.client.data_object.delete(
                class_name=self.class_name,
                uuid=obj_id
            )
        
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self.client is None:
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "is_initialized": False
            }
            
        result = self.client.query.aggregate(self.class_name).with_meta_count().do()
        count = result["data"]["Aggregate"][self.class_name][0]["meta"]["count"]
        
        return {
            "total_vectors": count,
            "dimension": self.dimension,
            "is_initialized": True
        } 