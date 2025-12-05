"""
Two-stage retrieval system: BM25 + Neural Re-ranking
"""
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch

class RetrievalSystem:
    """Two-stage retrieval system combining BM25 and neural re-ranking."""
    
    def __init__(self, passages):
        """
        Initialize retrieval system with passages.
        
        Args:
            passages: List of text passages to search over
        """
        self.passages = passages
        
        # Stage 1: BM25 - simple tokenization (split by spaces)
        self.tokenized = [p.lower().split() for p in passages]
        self.bm25 = BM25Okapi(self.tokenized)
        
        # Stage 2: Neural re-ranker
        print("Loading neural model (first time may take a minute)...")
        self.reranker = SentenceTransformer('all-MiniLM-L6-v2')
    
    def search(self, query, use_reranking=True, top_k=10):
        """
        Search for relevant passages using two-stage retrieval.
        
        Args:
            query: Search query string
            use_reranking: Whether to use neural re-ranking (default: True)
            top_k: Number of results to return
        
        Returns:
            List of (passage_text, score) tuples, sorted by relevance
        """
        # Stage 1: BM25 retrieval (get top 100)
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_100_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]
        
        # Get top 100 passages
        top_passages = [self.passages[i] for i in top_100_indices]
        
        # Stage 2: Neural re-ranking (if enabled)
        if use_reranking and len(top_passages) > 0:
            query_embedding = self.reranker.encode(query, convert_to_tensor=True)
            passage_embeddings = self.reranker.encode(top_passages, convert_to_tensor=True)
            similarities = util.cos_sim(query_embedding, passage_embeddings)[0]
            
            top_k_indices = torch.argsort(similarities, descending=True)[:top_k].cpu().tolist()
            results = [(top_passages[i], float(similarities[i])) for i in top_k_indices]
        else:
            # Just use BM25 results
            results = [(self.passages[top_100_indices[i]], scores[top_100_indices[i]]) 
                      for i in range(min(top_k, len(top_100_indices)))]
        
        return results

def load_data(filepath='data/corpus.json'):
    """Load passages from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get('passages', [])

