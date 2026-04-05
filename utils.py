# utils.py
# Helper functions for the Streamlit app

import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pinecone import Pinecone
from typing import List, Dict
from collections import defaultdict
import os

# Global variables to cache loaded models
embedding_model = None
cross_encoder = None
index = None
bm25_retriever = None

def load_models():
    """Load all models and initialize connections"""
    global embedding_model, cross_encoder, index, bm25_retriever
    
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    if cross_encoder is None:
        print("Loading cross-encoder...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    if index is None:
        print("Connecting to Pinecone...")
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("pakistan-history-rag")
    
    if bm25_retriever is None:
        print("Loading BM25 index...")
        import pickle
        with open('bm25_index.pkl', 'rb') as f:
            bm25_data = pickle.load(f)
        
        class SimpleBM25Retriever:
            def __init__(self, bm25_obj, chunks, tokenizer_func):
                self.bm25 = bm25_obj
                self.chunks = chunks
                self.tokenize = tokenizer_func
            
            def search(self, query, top_k=10):
                tokenized_query = self.tokenize(query)
                scores = self.bm25.get_scores(tokenized_query)
                top_indices = np.argsort(scores)[::-1][:top_k]
                
                results = []
                for idx in top_indices:
                    if scores[idx] > 0:
                        results.append({
                            'id': self.chunks[idx]['id'],
                            'text': self.chunks[idx]['text'],
                            'score': float(scores[idx])
                        })
                return results
        
        bm25_retriever = SimpleBM25Retriever(
            bm25_data['bm25'],
            bm25_data['chunks'],
            bm25_data['tokenizer']
        )
    
    return embedding_model, cross_encoder, index, bm25_retriever

def reciprocal_rank_fusion(semantic_results, bm25_results, k=60):
    """Fuse search results using RRF"""
    fusion_scores = defaultdict(float)
    result_data = {}
    
    for rank, result in enumerate(semantic_results, start=1):
        result_id = result['id']
        fusion_scores[result_id] += 1 / (k + rank)
        result_data[result_id] = result
    
    for rank, result in enumerate(bm25_results, start=1):
        result_id = result['id']
        fusion_scores[result_id] += 1 / (k + rank)
        if result_id not in result_data:
            result_data[result_id] = result
    
    sorted_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
    final_results = []
    for result_id, score in sorted_results:
        result = result_data[result_id]
        result['fusion_score'] = score
        final_results.append(result)
    
    return final_results

def hybrid_search(query, embedding_model, index, bm25_retriever, top_k=20):
    """Perform hybrid search"""
    query_embedding = embedding_model.encode(query)
    semantic_results_raw = index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    
    semantic_results = []
    for match in semantic_results_raw['matches']:
        semantic_results.append({
            'id': match['id'],
            'text': match['metadata']['text'],
            'score': match['score'],
            'metadata': match['metadata']
        })
    
    bm25_results = bm25_retriever.search(query, top_k=top_k)
    fused_results = reciprocal_rank_fusion(semantic_results, bm25_results)
    
    return fused_results

def rerank_results(query, results, cross_encoder):
    """Re-rank results using cross-encoder"""
    if not results:
        return results
    
    pairs = [(query, result['text']) for result in results]
    scores = cross_encoder.predict(pairs)
    
    for i, result in enumerate(results):
        result['rerank_score'] = float(scores[i])
    
    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)

def search_and_generate(query, embedding_model, cross_encoder, index, bm25_retriever, 
                        use_rerank=True, top_k=5):
    """Complete search and generation pipeline"""
    # Step 1: Hybrid search
    results = hybrid_search(query, embedding_model, index, bm25_retriever, top_k=20)
    
    # Step 2: Re-rank if requested
    if use_rerank:
        results = rerank_results(query, results, cross_encoder)
    
    # Step 3: Get top chunks
    top_chunks = results[:top_k]
    
    # Step 4: Prepare context
    context = "\n\n---\n\n".join([chunk['text'] for chunk in top_chunks])
    
    return {
        'context': context,
        'chunks': top_chunks,
        'num_chunks': len(top_chunks)
    }