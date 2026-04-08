
import pandas as pd
import numpy as np
import scipy.sparse as sp
import numpy as np
from math import log2
def dict_to_lists(train_seqs):
    """
    Convert a dictionary of lists to two separate lists: keys and values.
    """
    keys = []
    values = []
    for key, val_list in train_seqs.items():
        keys.append(key)
        values.append(val_list)
    return np.array(keys), np.array(values)
def compute_mrr(model, predict_fn, users, test_items_by_user, candidate_items, user2idx, item2idx):
    rr_list = []
    for u in users:
        # 1) score all candidates
        scores = [(i, predict_fn(model, u, i, user2idx, item2idx))
                  for i in candidate_items]
        # 2) sort by descending score
        ranked = [i for i,_ in sorted(scores, key=lambda x: -x[1])]
        # 3) find rank of first relevant item
        test_set = set(test_items_by_user[u])
        for rank, item in enumerate(ranked, start=1):
            if item in test_set:
                rr_list.append(1.0 / rank)
                break
        else:
            rr_list.append(0.0)
    return np.mean(rr_list)

def compute_ndcg_at_k(model, predict_fn, users, test_items_by_user,
                      candidate_items, user2idx, item2idx, K=10):
    ndcg_list = []
    for u in users:
        scores = [(i, predict_fn(model, u, i, user2idx, item2idx))
                  for i in candidate_items]
        ranked = [i for i,_ in sorted(scores, key=lambda x: -x[1])]
        dcg = 0.0
        for rank, item in enumerate(ranked[:K], start=1):
            if item in test_items_by_user[u]:
                dcg += 1.0 / log2(rank + 1)
        # ideal DCG: all test items at top
        n_rel = min(K, len(test_items_by_user[u]))
        idcg = sum(1.0 / log2(r+1) for r in range(1, n_rel+1))
        ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcg_list)





def per_user_last_split(
    df: pd.DataFrame,
    test_frac: float = 0.5,
    user_col: str = "Uid",
    time_col: str = "Timestamp"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort their rows by `time_col` ascending and
    put the last `max(1, int(n * test_frac))` rows into test,
    the rest into train.

    Args:
      df:         full interaction dataframe
      test_frac:  fraction of each user’s interactions to hold out
      user_col:   name of the user‐ID column
      time_col:   name of the datetime/sort column

    Returns:
      train, test  — two DataFrames with reset indices
    """
    train_parts = []
    test_parts  = []

    # make sure the time column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

    # split per user
    for uid, user_df in df.groupby(user_col, sort=False):
        user_df = user_df.sort_values(time_col)
        n = len(user_df)
        if n < 2:
            # not enough interactions → keep all in train
            train_parts.append(user_df)
            continue

        tsize   = max(1, int(n * test_frac))
        test_df = user_df.iloc[-tsize:]
        train_df= user_df.iloc[:-tsize]

        test_parts.append(test_df)
        train_parts.append(train_df)

    train = pd.concat(train_parts, ignore_index=True)
    test  = pd.concat(test_parts,  ignore_index=True)
    return train, test

#--------------------------------------------------------------------------------
# 2) Preprocess into R and mappings (unchanged)
#--------------------------------------------------------------------------------
def preprocess_data(df, train_df):
    users = df['Uid'].unique()
    items = df['Location ID'].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: j for j, i in enumerate(items)}
    n_users, n_items = len(users), len(items)

    R = sp.lil_matrix((n_users, n_items))
    ordered_list = []
    for _, row in train_df.iterrows():
        ordered_list.append(item2idx[row['Location ID']])
        R[user2idx[row['Uid']], item2idx[row['Location ID']]] = 1
    return R, user2idx, item2idx, n_users, n_items

