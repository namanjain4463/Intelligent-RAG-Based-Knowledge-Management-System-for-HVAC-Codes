"""Deterministic reciprocal-rank fusion, using section identity across retrievers."""
from copy import deepcopy

def fuse_rankings(rankings, top_k=10, constant=60):
    scores, candidates, methods = {}, {}, {}
    for ranking in rankings:
        seen = set()
        for rank, candidate in enumerate(ranking, 1):
            key = str(candidate['section']['number'])
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + rank)
            candidates.setdefault(key, deepcopy(candidate))
            methods.setdefault(key, set()).add(candidate.get('retrieval_method', 'unknown'))
    keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    result = []
    for key in keys:
        item = candidates[key]
        item['fusion_score'] = scores[key]
        item['retrieval_methods'] = sorted(methods[key])
        result.append(item)
    return result
