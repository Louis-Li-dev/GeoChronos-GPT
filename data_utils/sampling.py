
import pandas as pd
import scipy.sparse as sp
def select_east_coast(df):
    df = df.copy()
    east_coast = (
        df['Longitude'].between(-80, -67) &
        df['Latitude'].between(24, 50)
    )
    return df.loc[east_coast].reset_index(drop=True)
def preprocessing(df):
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = (
        df
        .sort_values('Timestamp')
        .drop_duplicates(subset=['Uid', 'Location ID'], keep='first')
    )
    return df
def get_sample(df, sample_size, max_element):
    df = df.copy().iloc[:sample_size]
    print(df.shape)
    new_df= df.groupby('Uid').filter(lambda x: len(x) \
         >= max_element).groupby('Uid', group_keys=False).head(max_element)
    return new_df

# def time_seq_process(df):
#     # 確保 Timestamp 是 datetime 格式
#     df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
#     # 依照 Uid 分組，並根據 Timestamp 排序後，彙整成 list
#     user_sequences = (
#         df.sort_values(['Uid', 'Timestamp'])          # 先對整體按 user 和時間排序
#         .groupby('Uid')['Location ID']              # 分組：每位使用者
#         .apply(list)                                # 每組轉成 list
#         .reset_index(name='location_sequence')      # Optional: 變成 DataFrame
#     )
#     return user_sequences

# def split_user_sequences(
#     df: pd.DataFrame,
#     test_frac: float=0.5,
#     user_col: str ='Uid',
#     time_col: str='Timestamp',
#     item_col: str='Location ID'
#     ):
#     df = df.copy()
#     df[time_col] = pd.to_datetime(df[time_col])

#     # 預設空字典：user → list of items
#     train_dt = {}
#     test_dt = {}

#     # 按使用者分組並排序
#     for uid, group in df.groupby(user_col):
#         # 根據時間排序
#         group = group.sort_values(time_col)
#         items = group[item_col].tolist()

#         # 切分 index
#         split_idx = int(len(items) * test_frac)
#         train_dt[uid] = items[:split_idx]
#         test_dt[uid] = items[split_idx:]

#     return train_dt, test_dt

def preprocess_data(df, train_df, test_df):
    # 1) build user/item mappings on the full df
    users = df['Uid'].unique()
    items = df['Location ID'].unique()
    user2idx = {u: i for i, u in enumerate(users)}
    item2idx = {i: j for j, i in enumerate(items)}
    n_users, n_items = len(users) + 1, len(items) + 1

    # helper to build R and ordered_list for any split
    def build_split(split_df):
        # group by user → list of itemIDs
        grp = split_df.groupby('Uid')["Location ID"].agg(list).reset_index()
        R = sp.lil_matrix((n_users, n_items), dtype=int)
        ordered = []
        for _, row in grp.iterrows():
            u = row['Uid']
            seq = [item2idx[i] for i in row['Location ID'] if i in item2idx]
            ordered.append(seq)
            R[user2idx[u], seq] = 1
        return R, ordered

    R_train, train_ordered_list = build_split(train_df)
    R_test,  test_ordered_list  = build_split(test_df)

    return (
        R_train,
        R_test,
        user2idx,
        item2idx,
        n_users,
        n_items,
        train_ordered_list,
        test_ordered_list
    )

def preprocess_seq(train_seqs,user2idx,item2idx, verbose=False):
    # inv_map = {v: k for k, v in item2idx.items()}
    # for k,v in item2idx.items():
    #     print(f'{k}:{v}')
    # print(f'size pf inv_map is {len(inv_map)}')
    for uid, seq in train_seqs.items():
        answer = []
        if verbose:
            print(f'size of {uid} is {len(seq)}')
        for i in seq:
            if i in item2idx:
                answer.append(item2idx[i])
            elif verbose:
                print(f'location {i} is not in inv_map')
        train_seqs[uid]=answer
    return train_seqs