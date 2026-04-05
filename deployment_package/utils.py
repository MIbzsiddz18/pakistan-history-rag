# utils.py
# Shared retrieval helper functions for the Streamlit app

import re
import numpy as np
from collections import defaultdict

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

def rerank_results(query, results, cross_encoder):
    """Re-rank results using cross-encoder.
    
    Accepts results where text may be at result['text'] or result['metadata']['text'].
    Returns all results sorted by rerank_score (caller slices to top_k).
    """
    if not results:
        return results

    def get_text(r):
        return r.get('text', r.get('metadata', {}).get('text', ''))

    pairs = [(query, get_text(result)) for result in results]
    scores = cross_encoder.predict(pairs)

    for i, result in enumerate(results):
        result['rerank_score'] = float(scores[i])

    return sorted(results, key=lambda x: x['rerank_score'], reverse=True)