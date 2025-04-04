from typing import Dict, Any, Tuple
import requests
from elasticsearch import Elasticsearch
import weaviate
import pinecone
import time

def validate_pinecone_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate Pinecone configuration and test connection."""
    try:
        # Initialize Pinecone
        pinecone.init(
            api_key=config.get("api_key"),
            environment=config.get("environment")
        )
        
        # Test index access
        index_name = config.get("index_name")
        if index_name not in pinecone.list_indexes():
            return False, f"Index '{index_name}' not found"
        
        # Test basic operations
        index = pinecone.Index(index_name)
        test_vector = [0.1] * 128  # 128-dimensional test vector
        
        # Test upsert
        start_time = time.time()
        index.upsert([(f"test_{time.time()}", test_vector)])
        upsert_time = time.time() - start_time
        
        # Test query
        start_time = time.time()
        results = index.query(test_vector, top_k=1)
        query_time = time.time() - start_time
        
        # Test delete
        start_time = time.time()
        index.delete([f"test_{time.time()}"])
        delete_time = time.time() - start_time
        
        return True, {
            "message": "Pinecone connection successful",
            "metrics": {
                "upsert_time": upsert_time,
                "query_time": query_time,
                "delete_time": delete_time
            }
        }
    except Exception as e:
        return False, str(e)

def validate_elasticsearch_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate Elasticsearch configuration and test connection."""
    try:
        # Initialize Elasticsearch client
        es = Elasticsearch(
            config.get("url"),
            basic_auth=(config.get("username"), config.get("password")),
            verify_certs=False  # For testing only
        )
        
        # Test cluster health
        health = es.cluster.health()
        if health["status"] not in ["green", "yellow"]:
            return False, f"Cluster health is {health['status']}"
        
        # Test index operations
        test_index = "test_validation"
        test_vector = [0.1] * 128
        
        # Test index creation and document insertion
        start_time = time.time()
        es.indices.create(index=test_index, ignore=400)
        es.index(
            index=test_index,
            document={
                "vector": test_vector,
                "timestamp": time.time()
            }
        )
        insert_time = time.time() - start_time
        
        # Test search
        start_time = time.time()
        es.search(
            index=test_index,
            query={
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                        "params": {"query_vector": test_vector}
                    }
                }
            }
        )
        search_time = time.time() - start_time
        
        # Cleanup
        es.indices.delete(index=test_index, ignore=[400, 404])
        
        return True, {
            "message": "Elasticsearch connection successful",
            "metrics": {
                "insert_time": insert_time,
                "search_time": search_time
            }
        }
    except Exception as e:
        return False, str(e)

def validate_weaviate_config(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate Weaviate configuration and test connection."""
    try:
        # Initialize Weaviate client
        client = weaviate.Client(
            url=config.get("url"),
            auth_client_secret=weaviate.AuthApiKey(api_key=config.get("api_key"))
        )
        
        # Test schema operations
        test_class = "TestValidation"
        test_vector = [0.1] * 128
        
        # Test class creation and object insertion
        start_time = time.time()
        client.schema.create_class({
            "class": test_class,
            "vectorizer": "none",
            "properties": [
                {"name": "test", "dataType": ["text"]}
            ]
        })
        client.data_object.create(
            data_object={"test": "validation"},
            class_name=test_class,
            vector=test_vector
        )
        insert_time = time.time() - start_time
        
        # Test search
        start_time = time.time()
        client.query.get(test_class, ["test"]).with_near_vector({
            "vector": test_vector
        }).with_limit(1).do()
        search_time = time.time() - start_time
        
        # Cleanup
        client.schema.delete_class(test_class)
        
        return True, {
            "message": "Weaviate connection successful",
            "metrics": {
                "insert_time": insert_time,
                "search_time": search_time
            }
        }
    except Exception as e:
        return False, str(e) 