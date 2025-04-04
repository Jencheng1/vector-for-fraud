from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from .vector_db.base import VectorDBInterface
from .vector_db.pinecone import PineconeDB
from .vector_db.weaviate import WeaviateDB
from .vector_db.faiss import FAISSDB
from .vector_db.chroma import ChromaDB
from .vector_db.elasticsearch import ElasticsearchDB
from .evaluation.metrics import calculate_metrics, evaluate_metrics
from .utils.validation import validate_pinecone_config, validate_elasticsearch_config, validate_weaviate_config
import time

app = FastAPI(title="Fraud RAG Vector DB Evaluation")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector databases
vector_dbs: Dict[str, VectorDBInterface] = {
    "pinecone": PineconeDB(),
    "weaviate": WeaviateDB(),
    "faiss": FAISSDB(),
    "chroma": ChromaDB(),
    "elasticsearch": ElasticsearchDB()
}

# Store database instances
db_instances = {}

class VectorRequest(BaseModel):
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]
    db_name: str

class SearchRequest(BaseModel):
    query_vector: List[float]
    k: int = 5
    db_name: str

class DBConfigRequest(BaseModel):
    db_name: str
    config: Dict[str, Any]

class InsertRequest(BaseModel):
    db_name: str
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]

class EvaluateRequest(BaseModel):
    db_name: str
    vectors: List[List[float]]
    metadata: List[Dict[str, Any]]

def get_db_instance(db_name: str) -> VectorDBInterface:
    """Get or create a database instance."""
    if db_name not in db_instances:
        if db_name == "pinecone":
            db_instances[db_name] = PineconeDB()
        elif db_name == "weaviate":
            db_instances[db_name] = WeaviateDB()
        elif db_name == "faiss":
            db_instances[db_name] = FAISSDB()
        elif db_name == "chroma":
            db_instances[db_name] = ChromaDB()
        elif db_name == "elasticsearch":
            db_instances[db_name] = ElasticsearchDB()
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported database: {db_name}")
    return db_instances[db_name]

@app.post("/validate/{db_name}")
async def validate_db_connection(db_name: str, config: Dict[str, Any]):
    """Validate connection to the specified database with provided configuration."""
    try:
        if db_name == "pinecone":
            success, result = validate_pinecone_config(config)
        elif db_name == "elasticsearch":
            success, result = validate_elasticsearch_config(config)
        elif db_name == "weaviate":
            success, result = validate_weaviate_config(config)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported database: {db_name}")
        
        if success:
            return result
        else:
            raise HTTPException(status_code=400, detail=result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/config")
async def configure_db(request: DBConfigRequest):
    """Configure a vector database with provided settings."""
    try:
        # Validate configuration first
        if request.db_name in ["pinecone", "elasticsearch", "weaviate"]:
            success, result = await validate_db_connection(request.db_name, request.config)
            if not success:
                raise HTTPException(status_code=400, detail=result)
        
        # Configure database if validation passes
        db = get_db_instance(request.db_name)
        db.configure(request.config)
        return {"status": "success", "message": f"{request.db_name} configured successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/test-connection/{db_name}")
async def test_connection(db_name: str):
    """Test connection to the specified database."""
    try:
        db = get_db_instance(db_name)
        # Try to perform a simple operation
        test_vector = np.random.rand(128).tolist()
        test_metadata = {"test": "connection"}
        
        # Test insertion
        start_time = time.time()
        db.insert([test_vector], [test_metadata])
        insert_time = time.time() - start_time
        
        # Test search
        start_time = time.time()
        results = db.search(test_vector, k=1)
        search_time = time.time() - start_time
        
        # Test deletion
        start_time = time.time()
        db.delete([test_metadata["id"]])
        delete_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": f"Successfully connected to {db_name}",
            "metrics": {
                "insert_time": insert_time,
                "search_time": search_time,
                "delete_time": delete_time
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/initialize/{db_name}")
async def initialize_db(db_name: str):
    """Initialize the specified vector database."""
    try:
        db = get_db_instance(db_name)
        db.initialize()
        return {"status": "success", "message": f"{db_name} initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/insert")
async def insert_data(request: InsertRequest):
    """Insert vectors and metadata into the specified database."""
    try:
        db = get_db_instance(request.db_name)
        start_time = time.time()
        db.insert(request.vectors, request.metadata)
        insertion_time = time.time() - start_time
        return {"status": "success", "insertion_time": insertion_time}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/evaluate")
async def evaluate_db(request: EvaluateRequest):
    """Evaluate performance metrics for the specified database."""
    try:
        db = get_db_instance(request.db_name)
        metrics = evaluate_metrics(db, request.vectors, request.metadata)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health_check():
    """Check the health of the API and connected databases."""
    health_status = {
        "api": "healthy",
        "databases": {}
    }
    
    for db_name in db_instances:
        try:
            db = db_instances[db_name]
            # Try a simple operation
            test_vector = np.random.rand(128).tolist()
            db.search(test_vector, k=1)
            health_status["databases"][db_name] = "healthy"
        except Exception as e:
            health_status["databases"][db_name] = f"unhealthy: {str(e)}"
    
    return health_status

@app.post("/search")
async def search_vectors(request: SearchRequest):
    if request.db_name not in vector_dbs:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        query_vector = np.array(request.query_vector)
        results = vector_dbs[request.db_name].search(query_vector, request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{db_name}")
async def get_stats(db_name: str):
    if db_name not in vector_dbs:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        stats = vector_dbs[db_name].get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 