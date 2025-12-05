"""
Evaluation metrics for retrieval system.
"""
import numpy as np
from typing import List, Set

def precision_at_k(relevant: Set[int], retrieved: List[int], k: int) -> float:
    """
    Calculate Precision@k.
    
    Args:
        relevant: Set of indices of relevant passages
        retrieved: List of retrieved passage indices (sorted by relevance)
        k: Number of top results to consider
        
    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_retrieved = len([idx for idx in top_k if idx in relevant])
    return relevant_retrieved / k

def recall_at_k(relevant: Set[int], retrieved: List[int], k: int) -> float:
    """
    Calculate Recall@k.
    
    Args:
        relevant: Set of indices of relevant passages
        retrieved: List of retrieved passage indices (sorted by relevance)
        k: Number of top results to consider
        
    Returns:
        Recall@k score
    """
    if len(relevant) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    relevant_retrieved = len([idx for idx in top_k if idx in relevant])
    return relevant_retrieved / len(relevant)

def ndcg_at_k(relevant: Set[int], retrieved: List[int], k: int) -> float:
    """
    Calculate nDCG@k (Normalized Discounted Cumulative Gain).
    
    Args:
        relevant: Set of indices of relevant passages
        retrieved: List of retrieved passage indices (sorted by relevance)
        k: Number of top results to consider
        
    Returns:
        nDCG@k score
    """
    if k == 0 or len(relevant) == 0:
        return 0.0
    
    top_k = retrieved[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, idx in enumerate(top_k):
        if idx in relevant:
            # Relevance is 1 if relevant, 0 otherwise
            rel = 1
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Calculate IDCG (ideal DCG)
    num_relevant = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg

def evaluate_retrieval(relevant: Set[int], retrieved: List[int]) -> dict:
    """
    Evaluate retrieval results using multiple metrics.
    
    Args:
        relevant: Set of indices of relevant passages
        retrieved: List of retrieved passage indices (sorted by relevance)
        
    Returns:
        Dictionary with evaluation metrics (Precision@5, Recall@10, nDCG@10)
    """
    return {
        'precision@5': precision_at_k(relevant, retrieved, 5),
        'precision@10': precision_at_k(relevant, retrieved, 10),
        'recall@10': recall_at_k(relevant, retrieved, 10),
        'ndcg@10': ndcg_at_k(relevant, retrieved, 10),
    }

