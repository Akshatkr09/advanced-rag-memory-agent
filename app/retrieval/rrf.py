"""Reciprocal Rank Fusion for combining independently ranked result lists."""
from collections import defaultdict

def reciprocal_rank_fusion(*ranked_lists, k: int = 60, id_key: str = "child_id"):
    """Fuse ranked lists while preserving metadata from every retrieval method.

    Each ranked list may contain dictionaries with ``id_key`` or plain string IDs.
    Repeated IDs accumulate RRF score and merge fields, so dense and sparse
    retrieval scores remain visible on the same fused result.
    """
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    scores = defaultdict(float)
    records = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_id = item if isinstance(item, str) else item.get(id_key)
            if not item_id:
                continue

            scores[item_id] += 1.0 / (k + rank)

            if item_id not in records:
                records[item_id] = {id_key: item_id}
            if not isinstance(item, str):
                # Merge rather than replace: preserve dense_score, sparse_score,
                # and any other metadata supplied by either retrieval method.
                records[item_id].update(item)

    fused = []
    for item_id, score in scores.items():
        record = dict(records[item_id])
        record["rrf_score"] = score
        fused.append(record)

    return sorted(fused, key=lambda item: item["rrf_score"], reverse=True)
