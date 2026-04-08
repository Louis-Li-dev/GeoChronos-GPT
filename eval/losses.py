import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

def topk(ranked_items, true_set, k):  
    rels = [1 if it in true_set else 0 for it in ranked_items]
    
    who = []
    for it in ranked_items:
        if it in true_set:
            who.append(it)
    
    tp = sum(rels[:k])
    topk_precision = tp/k
    topk_recall = tp/len(true_set)
    topk_accurarcy = tp > 0

    return {
        "hit":tp,
        "topk_precision":topk_precision,
        "topk_recall":topk_recall,
        "topk_accurarcy":topk_accurarcy,
        "who_get_touch":who
    }


def dcg_at_k(rels, k):
    """
    rels: list of relevance scores (0 or 1) in ranked order
    k:    cutoff
    """
    return sum(
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(rels[:k])
    )

def ndcg_at_k(ranked_items, true_set, k):
    """
    ranked_items : list of predicted item-IDs (length ≥ k)
    true_set     : full set of relevant item-IDs
    """
    rels = [1 if it in true_set else 0 for it in ranked_items[:k]]
    dcg  = dcg_at_k(rels, k)

    # ideal list: as many 1's as there are relevant items (capped at k)
    n_rels = min(len(true_set), k)
    ideal_rels = [1] * n_rels + [0] * (k - n_rels)
    idcg = dcg_at_k(ideal_rels, k)

    return dcg / idcg if idcg > 0 else 0.0


#--------------------------------------------------------------------------------
# 0) Setup
#--------------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def user_precision_worker(
    u, train_items, test_items, all_items, Ks, model, pred_fn, user2idx, item2idx
):
    seen = train_items.get(u, set())
    test = test_items[u]
    if not test:
        return {k: None for k in Ks}
    candidates = list(all_items - seen)
    if len(candidates) == 0:
        return {k: None for k in Ks}
    uidx = torch.tensor([user2idx[u]], dtype=torch.long)
    iidxs = torch.tensor([item2idx[i] for i in candidates], dtype=torch.long)
    try:
        with torch.no_grad():
            user_emb = model.user_emb(uidx)  # [1, d]
            item_emb = model.item_emb(iidxs) # [num_candidates, d]
            scores = (user_emb @ item_emb.T).squeeze(0).cpu().numpy()
    except Exception:
        scores = np.array([pred_fn(model, u, i, user2idx, item2idx) for i in candidates])
    max_k = max(Ks)
    # Use argpartition for fast top-K selection
    if len(scores) > max_k:
        topk_idx_unsorted = np.argpartition(-scores, max_k-1)[:max_k]
        topk_idx = topk_idx_unsorted[np.argsort(-scores[topk_idx_unsorted])]
    else:
        topk_idx = np.argsort(-scores)
    topk_items = [candidates[i] for i in topk_idx]
    test_set = set(test)
    hits = np.array([1 if item in test_set else 0 for item in topk_items])
    cum_hits = np.cumsum(hits)
    out = {}
    for k in Ks:
        if len(cum_hits) < k:
            out[k] = None  # Not enough candidates for this K
        else:
            out[k] = cum_hits[k-1] / k
    return out
def precision_at_k_joblib(
    model, pred_fn, R, user2idx, item2idx, train_df, test_df, Ks=[1,2,3,4,5], n_jobs=None, candidate_items=None
):
    train_items = train_df.groupby('Uid')['Location ID'].apply(set)
    test_items  = test_df.groupby('Uid')['Location ID'].apply(set)
    all_users = sorted(set(train_df['Uid']).union(set(test_df['Uid'])))
    # Use only candidate_items if provided
    all_items = candidate_items if candidate_items is not None else set(item2idx.keys())

    test_users = list(test_items.index)
    res = {k: [] for k in Ks}

    worker_args = [
        (u, train_items, test_items, all_items, Ks, model, pred_fn, user2idx, item2idx)
        for u in test_users
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(user_precision_worker)(*args) for args in tqdm(worker_args, desc="Precision@K users")
    )

    for out in results:
        for k in Ks:
            if out[k] is not None:
                res[k].append(out[k])

    summary = {k: np.mean(res[k]) if res[k] else 0.0 for k in Ks}
    print("\nPrecision@K summary:")
    for k in Ks:
        print(f"Precision@{k}: {summary[k]:.4f}")
    return pd.DataFrame({'K': Ks, 'Precision': [summary[k] for k in Ks]})


