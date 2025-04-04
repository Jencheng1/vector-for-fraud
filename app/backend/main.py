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
from .evaluation.metrics import calculate_metrics

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

@app.post("/initialize/{db_name}")
async def initialize_db(db_name: str, config: Optional[Dict[str, Any]] = None):
    if db_name not in vector_dbs:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        if config:
            # Update database configuration
            db = vector_dbs[db_name]
            if db_name == "elasticsearch":
                db.url = config.get("url", db.url)
                db.username = config.get("username", db.username)
                db.password = config.get("password", db.password)
            elif db_name == "weaviate":
                db.url = config.get("url", db.url)
                db.api_key = config.get("api_key", db.api_key)
        
        vector_dbs[db_name].initialize()
        return {"message": f"{db_name} initialized successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/insert")
async def insert_vectors(request: VectorRequest):
    if request.db_name not in vector_dbs:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        vectors = [np.array(v) for v in request.vectors]
        vector_dbs[request.db_name].insert_vectors(vectors, request.metadata)
        return {"message": "Vectors inserted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/evaluate")
async def evaluate_performance(request: VectorRequest):
    if request.db_name not in vector_dbs:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        metrics = calculate_metrics(
            vector_dbs[request.db_name],
            request.vectors,
            request.metadata
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/config")
async def update_db_config(request: DBConfigRequest):
    if request.db_name not in vector_dbs:
        raise HTTPException(status_code=404, detail="Database not found")
    try:
        db = vector_dbs[request.db_name]
        if request.db_name == "elasticsearch":
            db.url = request.config.get("url", db.url)
            db.username = request.config.get("username", db.username)
            db.password = request.config.get("password", db.password)
        elif request.db_name == "weaviate":
            db.url = request.config.get("url", db.url)
            db.api_key = request.config.get("api_key", db.api_key)
        return {"message": f"{request.db_name} configuration updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 