from typing import List, Dict, Any
import numpy as np
import time
from sklearn.metrics import precision_score, recall_score
from ..vector_db.base import VectorDBInterface

def calculate_metrics(
    vector_db: VectorDBInterface,
    vectors: List[List[float]],
    metadata: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Calculate performance metrics for the vector database.
    
    Args:
        vector_db: Vector database instance
        vectors: List of vectors to test
        metadata: List of metadata corresponding to vectors
    
    Returns:
        Dictionary containing performance metrics
    """
    metrics = {}
    
    # Measure insertion time
    start_time = time.time()
    vector_db.insert_vectors([np.array(v) for v in vectors], metadata)
    metrics['insertion_time'] = time.time() - start_time
    
    # Measure search time and accuracy
    search_times = []
    precisions = []
    recalls = []
    
    for i, query_vector in enumerate(vectors):
        # Get ground truth (assuming metadata contains 'is_fraud' field)
        ground_truth = metadata[i].get('is_fraud', 0)
        
        # Perform search
        start_time = time.time()
        results = vector_db.search(np.array(query_vector), k=5)
        search_time = time.time() - start_time
        search_times.append(search_time)
        
        # Calculate precision and recall
        predicted = [1 if r.get('is_fraud', 0) == 1 else 0 for r in results]
        if len(predicted) > 0:
            precisions.append(precision_score([ground_truth], [predicted[0]]))
            recalls.append(recall_score([ground_truth], [predicted[0]]))
    
    metrics['avg_search_time'] = np.mean(search_times)
    metrics['avg_precision'] = np.mean(precisions) if precisions else 0
    metrics['avg_recall'] = np.mean(recalls) if recalls else 0
    
    return metrics 