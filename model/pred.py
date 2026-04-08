import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import LightGCN
from torch_geometric.nn.models.lightgcn import BPRLoss
from typing import Union, Sequence, List, Optional
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from joblib import Parallel, delayed
from model.arrange import PlackettLuce
from typing import List, Sequence, Union, Optional, Dict, Any
import math
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    default_data_collator,
    TrainingArguments,
)
from torch.utils.data import Dataset as TorchDataset
from typing import List, Union, Optional
from torch.utils.data import DataLoader, TensorDataset





# -----------------------------------------------------------------------------
#  LightGCN w/ On-the-fly Negatives + optional PL re-ranker
# -----------------------------------------------------------------------------

class _ArrayDataset(TorchDataset):
    """Tiny wrapper so Hugging-Face Trainer can read NumPy / list IDs."""
    def __init__(self, ids: List[List[int]], pad_id: int):
        self.input_ids = ids
        self.pad_id = pad_id

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        attn = [0 if t == self.pad_id else 1 for t in ids]
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": ids,
        }


class GPT2Recommender:
    def __init__(
        self,
        model_name: str = "gpt2",
        pad_id: int = 0,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_id = pad_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)

    # ────────────────────────────────────────────────────────────
    #  TRAIN
    # ────────────────────────────────────────────────────────────
    def fit(
        self,
        id_sequences: Union[np.ndarray, List[List[int]]],
        *,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 5e-5,
        output_dir: str = "./gpt2_rec",
        fp16: bool = True,
    ):
        # 1) Normalise & cast
        if isinstance(id_sequences, np.ndarray):
            id_sequences = id_sequences.tolist()

        if not isinstance(id_sequences[0][0], (int, np.integer)):
            raise TypeError("IDs must be ints, not strings.")

        # 2) Resize vocab if needed
        max_id = max(max(seq) for seq in id_sequences)
        # if max_id >= self.model.config.vocab_size:
        self.model.resize_token_embeddings(max_id + 100)

        # 3) Build dataset
        ds = _ArrayDataset(id_sequences, self.pad_id)

        # 4) Trainer
        args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=fp16 and torch.cuda.is_available(),
            save_total_limit=2,
            logging_steps=100,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=ds,
            data_collator=default_data_collator,
        )
        self.trainer.train()
        return self  # chaining

    # ────────────────────────────────────────────────────────────
    #  INFERENCE
    # ────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def predict(
        self,
        prefix_ids: Union[np.ndarray, List[List[int]]],
        *,
        num_new_tokens: int = 10,
        do_sample: bool = False,
        device: Optional[str] = None,
        **generate_kwargs,
    ) -> List[List[int]]:
        """
        Generate continuations in ID space.

        Parameters
        ----------
        prefix_ids    : (batch, seq_len) int array / list
        num_new_tokens: how many tokens to append
        do_sample     : pass-through to `generate`
        device        : "cuda", "cpu", etc.  If None → use self.device

        Returns
        -------
        List[List[int]]  (batch of sequences, incl. prefix + newly-generated IDs)
        """
        # 1) resolve device -------------------------------------------------
        run_device = device or self.device
        if run_device != str(next(self.model.parameters()).device):
            self.model.to(run_device)

        # 2) normalise input -----------------------------------------------
        if isinstance(prefix_ids, np.ndarray):
            prefix_ids = prefix_ids.tolist()
        if isinstance(prefix_ids[0], int):          # single example
            prefix_ids = [prefix_ids]

        input_ids = torch.tensor(prefix_ids, dtype=torch.long, device=run_device)
        attention_mask = (input_ids != self.pad_id).long()

        # 3) autoregressive generation -------------------------------------
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=num_new_tokens,
            pad_token_id=self.pad_id,
            do_sample=do_sample,
            **generate_kwargs,
        )

        # 4) return plain Python lists of IDs ------------------------------
        return outputs.cpu().tolist()


class SelfAttention(nn.Module):
    """
    Scaled dot-product self-attention layer.
    """
    def __init__(self, dim_model, num_heads):
        super().__init__()
        assert dim_model % num_heads == 0, "dim_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(dim_model, dim_model * 3)
        self.out_proj = nn.Linear(dim_model, dim_model)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        scores = torch.einsum('bthd,bThd->bhtT', q, k) / self.scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        probs = F.softmax(scores, dim=-1)
        out = torch.einsum('bhtT,bThd->bthd', probs, v)
        out = out.reshape(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    """
    Simple two-layer feed-forward network.
    """
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderLayer(nn.Module):
    """
    One decoder layer with self-attention and feed-forward.
    """
    def __init__(self, dim_model, num_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SelfAttention(dim_model, num_heads)
        self.norm1 = nn.LayerNorm(dim_model)
        self.ff = FeedForward(dim_model, dim_ff, dropout)
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    """
    Minimal decoder-only Transformer with built-in training and prediction.
    """
    def __init__(
        self,
        vocab_size,
        dim_model=512,
        num_heads=8,
        num_layers=6,
        dim_ff=2048,
        max_seq_len=512,
        dropout=0.1,
        device=None
    ):
        super().__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_emb = nn.Embedding(vocab_size, dim_model)
        self.pos_emb = nn.Embedding(max_seq_len, dim_model)
        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, num_heads, dim_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim_model)
        self.output_proj = nn.Linear(dim_model, vocab_size, bias=False)
        self.to(self.device)

    def forward(self, input_ids):
        """
        input_ids: (batch, seq_len)
        returns logits: (batch, seq_len, vocab_size)
        """
        B, T = input_ids.shape
        x = self.token_emb(input_ids.to(self.device))
        pos = torch.arange(T, device=self.device)
        x = x + self.pos_emb(pos).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.output_proj(x)

    def fit(
        self,
        train_x,
        train_y,
        epochs=10,
        batch_size=32,
        lr=1e-3,
        verbose=True,
        print_batch_freq: int = 0
    ):
        """
        Train on (train_x, train_y) pairs of token IDs.

        train_x, train_y: torch.Tensor of shape (N, L)
        verbose: print epoch-level stats if True
        print_batch_freq: print batch loss every `print_batch_freq` batches (0 to disable)
        """
        train_x = train_x.to(self.device).long()
        train_y = train_y.to(self.device).long()
        ds = TensorDataset(train_x, train_y)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.train()

        for ep in range(1, epochs + 1):
            total_loss = 0.0
            for idx, (bx, by) in enumerate(loader, start=1):
                opt.zero_grad()
                logits = self(bx)  # (B, L, V)
                B, L, V = logits.shape
                loss = loss_fn(logits.view(-1, V), by.view(-1))
                loss.backward()
                opt.step()
                total_loss += loss.item() * B

                if print_batch_freq and idx % print_batch_freq == 0:
                    print(f"Epoch {ep} | Batch {idx}/{len(loader)} | batch_loss: {loss.item():.4f}")

            avg = total_loss / len(ds)
            if verbose:
                print(f"Epoch {ep}/{epochs} — avg_loss: {avg:.4f}")
        self.eval()


    def predict(self, inputs: Union[torch.Tensor, list, np.ndarray], slide: bool = True):
        """
        Generate predictions using sliding-window if needed.

        inputs: Tensor or array of shape (batch, T) or (T,)   
        slide: if True and T > max_seq_len, apply sliding window.

        Returns:
          if slide and T>max_seq_len:
              Tensor of shape (batch, n_windows, max_seq_len)
          else:
              Tensor of shape (batch, seq_len)
        """
        # Prepare input tensor
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, dtype=torch.long)
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device).long()

        B, T = inputs.shape
        L = min(T, self.max_seq_len)

        if slide and T > self.max_seq_len:
            n_windows = T - self.max_seq_len + 1
            preds = []
            for i in range(n_windows):
                window = inputs[:, i : i + self.max_seq_len]
                with torch.no_grad():
                    logits = self(window)
                    batch_preds = torch.argmax(logits, dim=-1)
                preds.append(batch_preds)
            # stack: (n_windows, B, L) -> (B, n_windows, L)
            preds = torch.stack(preds, dim=0).permute(1, 0, 2)
            return preds.cpu()
        else:
            inp = inputs[:, :L]
            with torch.no_grad():
                logits = self(inp)
                preds = torch.argmax(logits, dim=-1)
            return preds.cpu()  # (batch, L)

class LightGCNModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        layers: int = 3,
        emb_dim: int = 64,
        device: torch.device = None
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.net = LightGCN(
            num_nodes     = n_users + n_items,
            embedding_dim = emb_dim,
            num_layers    = layers,
            alpha         = 0.5
        )
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def forward(self, edge_index):
        all_emb = self.net(edge_index)
        return all_emb[:self.n_users], all_emb[self.n_users:]

    def fit(
        self,
        pos_u: torch.Tensor,
        pos_i: torch.Tensor,
        edge_index: torch.Tensor,
        epochs: int = 10,
        lr: float = 1e-2
    ):
        self.train()
        bpr        = BPRLoss().to(self.device)
        optimizer  = optim.Adam(self.parameters(), lr=lr)
        pos_u      = pos_u.to(self.device)
        pos_i      = pos_i.to(self.device)
        edge_index = edge_index.to(self.device)

        self._edge_index = edge_index
        for ep in range(1, epochs+1):
            optimizer.zero_grad()
            u_emb, i_emb = self(edge_index)
            neg_i = torch.randint(0, self.n_items, size=pos_i.size(), device=self.device)
            loss = bpr(u_emb[pos_u], i_emb[pos_i], i_emb[neg_i])
            loss.backward()
            optimizer.step()
            print(f"[GCN] Epoch {ep}/{epochs} — loss: {loss.item():.4f}")

        # cache final embeddings
        self.eval()
        with torch.no_grad():
            u_emb, i_emb = self(self._edge_index)
        self._u_emb = u_emb
        self._i_emb = i_emb

    def predict(
        self,
        u_id: int,
        i_ids: Union[int, Sequence[int]],
        user2idx: dict,
        item2idx: dict,
        reorder_model: Optional['PlackettLuce'] = None,
        k: Optional[int] = None
    ) -> Union[float, List[int]]:
        if not hasattr(self, '_u_emb'):
            raise RuntimeError("Call .fit(...) before .predict()")

        u_idx = user2idx[u_id]

        def _score(i_id):
            i_idx = item2idx[i_id]
            score = (self._u_emb[u_idx] * self._i_emb[i_idx]).sum()
            return float(torch.sigmoid(score))

        if isinstance(i_ids, int):
            return _score(i_ids)

        scored = {i: _score(i) for i in i_ids}
        ranked = sorted(i_ids, key=lambda x: scored[x], reverse=True)

        if reorder_model is not None:
            if k is None:
                raise ValueError("You must provide k when using reorder_model")
            topk = ranked[:k]
            numeric = [item2idx[i] for i in topk]
            reordered = reorder_model.predict(numeric)
            inv_map = {v: k for k, v in item2idx.items()}
            ranked = [inv_map[n] for n in reordered]

        return ranked



# -----------------------------------------------------------------------------
#  Matrix Factorization w/ Neg Sampling + optional PL re-ranker
# -----------------------------------------------------------------------------
class MFCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 50,
        device: torch.device = None
    ):
        super().__init__()
        self.U = nn.Embedding(n_users, emb_dim)
        self.I = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.U.weight, std=0.01)
        nn.init.normal_(self.I.weight, std=0.01)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def forward(self, u: torch.LongTensor, i: torch.LongTensor):
        return torch.sigmoid((self.U(u) * self.I(i)).sum(dim=1))

    def fit(
        self,
        train_df,
        user2idx: dict,
        item2idx: dict,
        R: np.ndarray,
        epochs: int = 10,
        lr: float = 1e-2,
        batch_size: int = 256,
        loc_col = "Location ID",
        user_col = 'Uid'
    ):
        pos = [(user2idx[u], item2idx[i]) for u,i in zip(train_df[user_col], train_df[loc_col])]
        neg = []
        for u,i in pos:
            j = np.random.randint(0, R.shape[1])
            while R[u, j] == 1:
                j = np.random.randint(0, R.shape[1])
            neg.append((u, j))

        us  = torch.tensor([u for u,i in pos]+[u for u,j in neg], dtype=torch.long)
        is_ = torch.tensor([i for u,i in pos]+[j for u,j in neg], dtype=torch.long)
        lbl = torch.tensor([1.0]*len(pos)+[0.0]*len(neg), dtype=torch.float)

        ds     = TensorDataset(us, is_, lbl)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        opt    = optim.Adam(self.parameters(), lr=lr)
        loss_fn= nn.BCELoss()
        self.train()
        for ep in range(1, epochs+1):
            total=0.0
            for ub, ib, yb in loader:
                ub, ib, yb = [t.to(self.device) for t in (ub, ib, yb)]
                opt.zero_grad()
                loss = loss_fn(self(ub, ib), yb)
                loss.backward()
                opt.step()
                total += loss.item()*ub.size(0)
            print(f"[MF] Epoch {ep}/{epochs} — loss: {total/len(ds):.4f}")

    def predict(
        self,
        u_id: int,
        i_ids: Union[int, Sequence[int]],
        user2idx: dict,
        item2idx: dict,
        reorder_model: Optional['PlackettLuce'] = None,
        k: Optional[int] = None
    ) -> Union[float, List[int]]:
        self.eval()
        u_idx = user2idx[u_id]

        def _score(i_id):
            i_idx = item2idx[i_id]
            u = torch.tensor([u_idx], dtype=torch.long, device=self.device)
            i = torch.tensor([i_idx], dtype=torch.long, device=self.device)
            with torch.no_grad():
                return float(self(u, i))

        if isinstance(i_ids, int):
            return _score(i_ids)

        scored = {i: _score(i) for i in i_ids}
        ranked = sorted(i_ids, key=lambda x: scored[x], reverse=True)

        if reorder_model is not None:
            if k is None:
                raise ValueError("You must provide k when using reorder_model")
            topk = ranked[:k]
            numeric = [item2idx[i] for i in topk]
            reordered = reorder_model.predict(numeric)
            inv_map = {v: k for k, v in item2idx.items()}
            ranked = [inv_map[n] for n in reordered]

        return ranked

class CF_user(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
    ):
        super().__init__()
        self.Function
        self.n_users = n_users
        self.n_items = n_items
        
    def fit(self,X):
        self.user_similarity = cosine_similarity(X)

    def predict(self,target_user,X):
        # 預測使用者 0 對所有商品的興趣
        target_user = 0
        sim_scores = self.user_similarity[target_user]        # 使用者 0 與其他使用者的相似度
        pred = sim_scores @ X.toarray()                  # 加權求和
        pred = pred / sim_scores.sum()                   # Normalize

        return pred
        
    def pairwise_cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        eps=1e-8
    ):
        """
        計算兩個矩陣之間所有成對 cosine 相似度
        a: (n, d)
        b: (m, d)
        return: (n, m) 相似度矩陣
        """
        # 正規化兩個矩陣的每一行（L2）
        a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
        b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
        
        # 內積就是 cosine similarity（已經正規化）
        return a_norm @ b_norm.T
        
class CF_item(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
    ):
        super().__init__()
        self.Function
        self.n_users = n_users
        self.n_items = n_items
        
    
    def fit(self,X):
        self.item_similarity = cosine_similarity(X.T)

    def predict(self,target_user,X):
        # 預測使用者 0 對所有商品的興趣
        
        user_interactions = X[target_user].toarray().flatten()
        pred = self.item_similarity @ user_interactions
        pred = pred / self.item_similarity.sum(axis=1)
        return pred
        
    def pairwise_cosine_similarity(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        eps=1e-8
    ):
        """
        計算兩個矩陣之間所有成對 cosine 相似度
        a: (n, d)
        b: (m, d)
        return: (n, m) 相似度矩陣
        """
        # 正規化兩個矩陣的每一行（L2）
        a_norm = a / (a.norm(dim=1, keepdim=True) + eps)
        b_norm = b / (b.norm(dim=1, keepdim=True) + eps)
        
        # 內積就是 cosine similarity（已經正規化）
        return a_norm @ b_norm.T



class CFUser(nn.Module):
    """
    User‐based CF (vectorized candidate‐scoring + top‐K via NumPy).
    """
    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.user_similarity: Optional[np.ndarray] = None  # (n_users x n_users)
        self.R_train: Optional[csr_matrix] = None

    def fit(self, R: Union[csr_matrix, np.ndarray]):
        # 1) Ensure CSR
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        self.R_train = R

        # 2) Compute user-user cosine similarity
        sim = cosine_similarity(R, dense_output=True)  # (n_users x n_users)
        np.fill_diagonal(sim, 0.0)
        self.user_similarity = sim

    def predict(
        self,
        u_id: Any,
        i_ids: Sequence[Any],
        user2idx: Dict[Any, int],
        item2idx: Dict[Any, int],
        reorder_model: Optional['PlackettLuce'] = None,
        k: Optional[int] = None
    ) -> Union[float, List[Any]]:
        """
        Vectorized “score all candidates in one shot” implementation.

        - If `i_ids` is a single int, we return one float.  
        - Otherwise, we treat `i_ids` as a list of original item‐IDs and:
            1) Compute r_hat_all = (R_train.T dot sim_scores) / sum(sim_scores)
            2) Gather the subset of r_hat_all corresponding to all `i_ids` (in one NumPy slice).
            3) Arg‐partition or argsort that small array to get top‐K or full sort.
            4) Convert internal indices back to original item‐IDs.
        """
        if self.user_similarity is None or self.R_train is None:
            raise RuntimeError("Call .fit(R_train) before .predict()")

        u_idx = user2idx[u_id]
        sim_scores = self.user_similarity[u_idx]        # shape=(n_users,)

        # -------- Single‐item scoring path --------
        if isinstance(i_ids, int):
            i_idx = item2idx[i_ids]
            denom = sim_scores.sum()
            if denom == 0:
                return 0.0
            # Equivalent to (sim_scores @ R_train[:, i_idx]) / denom
            weighted_sum = float(sim_scores @ self.R_train[:, i_idx].toarray().flatten())
            return weighted_sum / denom

        # -------- Multi‐item: vectorized scoring --------
        # 1) If sum(sim_scores)==0, all r_hat_all = 0
        denom = sim_scores.sum()
        if denom == 0:
            r_hat_all = np.zeros(self.n_items, dtype=np.float32)
        else:
            # R_train.T is (n_items × n_users), sim_scores is (n_users,).
            weighted_sum = self.R_train.T.dot(sim_scores)   # shape=(n_items,)
            r_hat_all = (weighted_sum / denom).astype(np.float32)

        # 2) Convert list of original IDs → NumPy array of internal indices
        #    Suppose i_ids = [orig1, orig2, orig3, ...]
        #    Let idxs = [item2idx[orig1], item2idx[orig2], ... ]
        idxs = np.array([item2idx[orig] for orig in i_ids], dtype=np.int32)

        # 3) Extract the scores of all candidates in one vector
        candidate_scores = r_hat_all[idxs]  # shape = (len(i_ids),)

        # 4) If the user did *not* request PL re‐ranking, we just need top K or sorted
        if reorder_model is None or k is None:
            if k is None or k >= len(idxs):
                # We want *all* candidates, fully sorted descending
                order = np.argsort(-candidate_scores)  # descending
                sorted_internal_idxs = idxs[order]     # these are internal indices
                return [   # map back to original IDs
                    list(i_ids)[pos]  # because i_ids[pos] corresponds to idxs[pos]
                    for pos in order
                ]
            else:
                # We only need the top‐k. Use argpartition for O(n) instead of O(n log n)
                topk_part = np.argpartition(-candidate_scores, k - 1)[:k]
                topk_internal = idxs[topk_part]                      # internal indices
                # But argpartition leaves the topk unsorted; now sort these k by descending score
                topk_scores = candidate_scores[topk_part]
                sorted_k = np.argsort(-topk_scores)  # descending among the k
                final_k_int = topk_internal[sorted_k]
                return [list(i_ids)[   # convert back to original IDs
                    # We need the original ID corresponding to internal idx final_k_int[j]:
                    # find its position in idxs. We can store a reverse‐mapping array:
                    # But easier: keep i_ids and idxs aligned: idxs[j] → i_ids[j].
                    np.where(idxs == final_k_int[j])[0][0]
                ] for j in range(len(sorted_k))]

        # 5) If PL + k is provided, we only need the top‐k internal indices
        #    (we’ll hand them to PL for re‐ranking)
        if k >= len(idxs):
            # If k >= #candidates, just sort all
            order = np.argsort(-candidate_scores)
            topk_internal = idxs[order]
        else:
            # argpartition to grab the k largest
            topk_part = np.argpartition(-candidate_scores, k - 1)[:k]
            topk_internal_unsorted = idxs[topk_part]
            topk_scores = candidate_scores[topk_part]
            sorted_k = np.argsort(-topk_scores)
            topk_internal = topk_internal_unsorted[sorted_k]

        # 6) Now hand those k internal IDs to PL.predict(...)
        boosted_internal = reorder_model.predict(list(topk_internal))
        # boosted_internal is a list of length k of internal indices, in PL‐order

        # 7) Map PL‐ordered internal indices back to original IDs
        inv_map = {v: k for k, v in item2idx.items()}
        boosted_original = [inv_map[int(internal_idx)] for internal_idx in boosted_internal]

        return boosted_original
class CFItem(nn.Module):
    """
    Item‐based Collaborative Filtering (vectorized).
    """
    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.item_similarity: Optional[np.ndarray] = None  # (n_items x n_items)
        self.R_train: Optional[csr_matrix] = None

    def fit(self, R: Union[csr_matrix, np.ndarray]):
        if not isinstance(R, csr_matrix):
            R = csr_matrix(R)
        self.R_train = R

        sim = cosine_similarity(R.T, dense_output=True)   # (n_items x n_items)
        np.fill_diagonal(sim, 0.0)
        self.item_similarity = sim

    def predict(
        self,
        u_id: Any,
        i_ids: Union[int, Sequence[Any]],
        user2idx: Dict[Any, int],
        item2idx: Dict[Any, int],
        reorder_model: Optional['PlackettLuce'] = None,
        k: Optional[int] = None
    ) -> Union[float, List[Any]]:
        """
        If i_ids is int: return float score for that one item.
        If i_ids is a list:
          1) Compute u_profile = R_train[u_idx, :].toarray().flatten()
          2) Compute numerator = item_similarity.dot(u_profile)  # (n_items,)
             denom = item_similarity.sum(axis=1)                 # (n_items,)
             r_hat_all = numerator / denom_safe
          3) Vectorized slice of r_hat_all at all internal indices for i_ids.
          4) argsort/argpartition that small array to get top‐K or full sort.
          5) (Optional) PL re‐rank top‐K.
        """
        if self.item_similarity is None or self.R_train is None:
            raise RuntimeError("Call .fit(R_train) before .predict()")

        u_idx = user2idx[u_id]

        # Single‐item path
        if isinstance(i_ids, int):
            i_idx = item2idx[i_ids]
            u_profile = self.R_train.getrow(u_idx).toarray().flatten()  # (n_items,)
            sim_i = self.item_similarity[i_idx]                        # (n_items,)
            denom = sim_i.sum()
            if denom == 0:
                return 0.0
            return float((sim_i * u_profile).sum() / denom)

        # Multi‐item: compute r_hat_all once
        u_profile = self.R_train.getrow(u_idx).toarray().flatten()      # shape=(n_items,)
        numerator = self.item_similarity.dot(u_profile)                # shape=(n_items,)
        denom = self.item_similarity.sum(axis=1)                       # shape=(n_items,)
        denom_safe = np.where(denom == 0, 1.0, denom)
        r_hat_all = (numerator / denom_safe).astype(np.float32)

        # Vectorize candidate scoring
        idxs = np.array([item2idx[orig] for orig in i_ids], dtype=np.int32)
        candidate_scores = r_hat_all[idxs]  # shape=(len(i_ids),)

        # No PL or no k → fully sort or top‐k sort
        if reorder_model is None or k is None:
            if k is None or k >= len(idxs):
                order = np.argsort(-candidate_scores)
                return [i_ids[pos] for pos in order]
            else:
                part = np.argpartition(-candidate_scores, k - 1)[:k]
                topk_internal = idxs[part]
                topk_scores = candidate_scores[part]
                sorted_k = np.argsort(-topk_scores)
                final_int = topk_internal[sorted_k]
                return [i_ids[np.where(idxs == final_int[j])[0][0]] for j in range(len(sorted_k))]

        # PL + k provided
        if k >= len(idxs):
            order = np.argsort(-candidate_scores)
            topk_internal = idxs[order]
        else:
            part = np.argpartition(-candidate_scores, k - 1)[:k]
            topk_unsorted = idxs[part]
            topk_scores = candidate_scores[part]
            sorted_k = np.argsort(-topk_scores)
            topk_internal = topk_unsorted[sorted_k]

        boosted_internal = reorder_model.predict(list(topk_internal))
        inv_map = {v: k for k, v in item2idx.items()}
        boosted_original = [inv_map[int(idx)] for idx in boosted_internal[:k]]
        return boosted_original


# ----------------------------------------------------------------------------- 
# 1) Pre‐compute version of CFUser with threading‐backend support
# -----------------------------------------------------------------------------
class CFUserPrecomputed(nn.Module):
    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.user_similarity: Optional[np.ndarray] = None
        self.R_train: Optional[csr_matrix] = None
        self.user_full_scores: Optional[np.ndarray] = None
        self.user_sorted_items: Optional[List[np.ndarray]] = None

    def fit(
        self,
        R: csr_matrix,
        n_jobs: int = 1,
        use_tqdm: bool = False
    ):
        """
        Args:
          R: CSR sparse matrix (n_users x n_items).
          n_jobs: number of parallel workers for per-user computation (threads if >1).
          use_tqdm: if True, show a tqdm progress bar.
        """
        if not isspmatrix_csr(R):
            raise ValueError("R must be a CSR sparse matrix")

        # 1) Compute and store R_train & user–user similarity
        self.R_train = R
        sim = cosine_similarity(R, dense_output=True)  # (n_users x n_users)
        np.fill_diagonal(sim, 0.0)
        self.user_similarity = sim

        # Allocate storage
        self.user_full_scores = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.user_sorted_items = [None] * self.n_users

        # Worker function for one user
        def _compute_for_user(u_idx: int):
            sim_scores = sim[u_idx]
            denom = sim_scores.sum()
            if denom == 0:
                r_hat = np.zeros(self.n_items, dtype=np.float32)
            else:
                weighted = R.T.dot(sim_scores)                # shape = (n_items,)
                r_hat = (weighted / denom).astype(np.float32)
            sorted_internal = np.argsort(-r_hat)              # descending
            return u_idx, r_hat, sorted_internal

        user_indices = list(range(self.n_users))
        if use_tqdm:
            user_iter = tqdm(user_indices, desc="CFUser⇒Precompute")
        else:
            user_iter = user_indices

        # Use threading backend to avoid pickling large arrays
        if n_jobs == 1:
            # Single‐threaded
            for u_idx in user_iter:
                idx, r_hat, sorted_internal = _compute_for_user(u_idx)
                self.user_full_scores[idx] = r_hat
                self.user_sorted_items[idx] = sorted_internal
        else:
            # Multi‐threaded (no pickling of R or sim required)
            results = Parallel(
                n_jobs=n_jobs,
                backend="threading"
            )(
                delayed(_compute_for_user)(u_idx) for u_idx in user_iter
            )
            for idx, r_hat, sorted_internal in results:
                self.user_full_scores[idx] = r_hat
                self.user_sorted_items[idx] = sorted_internal

    def predict(
        self,
        u_id: Any,
        i_ids: List[Any],
        user2idx: Dict[Any, int],
        idx2item: Dict[int, Any],
        item2idx: Dict[Any, int],
        reorder_model: Optional['PlackettLuce'] = None,
        k: Optional[int] = None
    ) -> List[Any]:
        if self.user_full_scores is None or self.user_sorted_items is None:
            raise RuntimeError("Call .fit(...) first")

        u_idx = user2idx[u_id]
        full_sorted = self.user_sorted_items[u_idx]  # array of length n_items

        candidate_set = set(item2idx[orig] for orig in i_ids)
        topk_internals = []

        if k is None:
            # return all candidates in descending order
            for internal_i in full_sorted:
                if internal_i in candidate_set:
                    topk_internals.append(internal_i)
        else:
            # stop once we have k
            for internal_i in full_sorted:
                if internal_i in candidate_set:
                    topk_internals.append(internal_i)
                    if len(topk_internals) >= k:
                        break

        if reorder_model is None or k is None:
            return [idx2item[ii] for ii in topk_internals]

        boosted_internal = reorder_model.predict(topk_internals)
        inv_map = {v: k for k, v in item2idx.items()}
        boosted_original = [inv_map[int(ii)] for ii in boosted_internal[:k]]
        return boosted_original


# ----------------------------------------------------------------------------- 
# 2) Pre‐compute version of CFItem with threading‐backend support 
# -----------------------------------------------------------------------------
class CFItemPrecomputed(nn.Module):
    def __init__(self, n_users: int, n_items: int):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.item_similarity: Optional[np.ndarray] = None
        self.R_train: Optional[csr_matrix] = None

        self.user_item_scores: Optional[np.ndarray] = None
        self.user_item_sorted: Optional[List[np.ndarray]] = None

    def fit(
        self,
        R: csr_matrix,
        n_jobs: int = 1,
        use_tqdm: bool = False
    ):
        """
        Args:
          R: CSR sparse matrix (n_users x n_items).
          n_jobs: number of parallel workers (threads) for per-user computation.
          use_tqdm: if True, show a tqdm progress bar.
        """
        if not isspmatrix_csr(R):
            raise ValueError("R must be a CSR sparse matrix")

        self.R_train = R
        sim = cosine_similarity(R.T, dense_output=True)  # (n_items x n_items)
        np.fill_diagonal(sim, 0.0)
        self.item_similarity = sim

        self.user_item_scores = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        self.user_item_sorted = [None] * self.n_users

        def _compute_for_user(u_idx: int):
            u_profile = R.getrow(u_idx).toarray().flatten()    # (n_items,)
            numerator = sim.dot(u_profile)                     # (n_items,)
            denom = sim.sum(axis=1)                            # (n_items,)
            denom_safe = np.where(denom == 0, 1.0, denom)
            r_hat = (numerator / denom_safe).astype(np.float32)
            sorted_internal = np.argsort(-r_hat)
            return u_idx, r_hat, sorted_internal

        user_indices = list(range(self.n_users))
        if use_tqdm:
            user_iter = tqdm(user_indices, desc="CFItem⇒Precompute")
        else:
            user_iter = user_indices

        if n_jobs == 1:
            for u_idx in user_iter:
                idx, r_hat, sorted_internal = _compute_for_user(u_idx)
                self.user_item_scores[idx] = r_hat
                self.user_item_sorted[idx] = sorted_internal
        else:
            results = Parallel(
                n_jobs=n_jobs,
                backend="threading"
            )(
                delayed(_compute_for_user)(u_idx) for u_idx in user_iter
            )
            for idx, r_hat, sorted_internal in results:
                self.user_item_scores[idx] = r_hat
                self.user_item_sorted[idx] = sorted_internal

    def predict(
        self,
        u_id: Any,
        i_ids: List[Any],
        user2idx: Dict[Any, int],
        idx2item: Dict[int, Any],
        item2idx: Dict[Any, int],
        reorder_model: Optional['PlackettLuce'] = None,
        k: Optional[int] = None
    ) -> List[Any]:
        if self.user_item_scores is None or self.user_item_sorted is None:
            raise RuntimeError("Call .fit(...) first")

        u_idx = user2idx[u_id]
        full_sorted = self.user_item_sorted[u_idx]

        candidate_set = set(item2idx[orig] for orig in i_ids)
        topk_internals = []

        if k is None:
            for internal_i in full_sorted:
                if internal_i in candidate_set:
                    topk_internals.append(internal_i)
        else:
            for internal_i in full_sorted:
                if internal_i in candidate_set:
                    topk_internals.append(internal_i)
                    if len(topk_internals) >= k:
                        break

        if reorder_model is None or k is None:
            return [idx2item[ii] for ii in topk_internals]

        boosted_internal = reorder_model.predict(topk_internals)
        inv_map = {v: k for k, v in item2idx.items()}
        boosted_original = [inv_map[int(ii)] for ii in boosted_internal[:k]]
        return boosted_original
    
class AttentionRecommender(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        layers=1,
        device='cpu'
        ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.layers = layers

        self.item_embedding = nn.Embedding(n_items, 64)
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.fc = nn.Linear(64, n_items)
        
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.to(self.device)

    def forward(
        self,
        seq_tensor
        ):
        # seq_tensor: (batch_size, seq_len)
        x = self.item_embedding(seq_tensor)  # (B, L, D)
        attn_out, _ = self.attn(x, x, x)     # Self-attention
        pooled = attn_out.mean(dim=1)        # Pool over sequence
        scores = self.fc(pooled)             # (B, n_items)
        return scores

    def fit(
        self,
        train_dt,
        epochs=10,
        lr=1e-3
        ):
        model = self
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Build train dataset
        data = []
        for uid, seq in train_dt.items():
            if len(seq) < 2: continue
            data.append((torch.tensor(seq[:-1]), seq[-1]))  # input seq, target item

        for epoch in range(epochs):
            total_loss = 0
            for seq, target in data:
                seq = seq.to(self.device).unsqueeze(0)  # (1, L)
                target = torch.tensor([target], device=self.device)  # (1,)
                optimizer.zero_grad()
                logits = model(seq)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    def predict(
        self,
        u_id: int,
        i_ids: Union[int, Sequence[int]],
        test_dt,
        user2idx: dict,
        item2idx: dict,
        k: Optional[int] = None
        ):
        self.eval()
        results = []
        with torch.no_grad():
            for uid, seq in test_dt.items():
                if len(seq) == 0:
                    results.append([])  # No sequence, no prediction
                    continue
                seq_tensor = torch.tensor(seq, device=self.device).unsqueeze(0)  # (1, L)
                logits = self.forward(seq_tensor)  # (1, n_items)
                top_items = torch.topk(logits, k=10, dim=1).indices.squeeze(0).tolist()
                results.append(top_items)
        return results
    
    def predict_one(
        self,
        u_id: int,
        sequence,
        top_k=10
        ):
        self.eval()
        with torch.no_grad():
            if not sequence:
                return []  # 空序列回傳空結果

        seq_tensor = torch.tensor(sequence, device=self.device).unsqueeze(0)  # (1, L)
        logits = self.forward(seq_tensor)  # (1, n_items)
        top_items = torch.topk(logits, k=top_k, dim=1).indices.squeeze(0).tolist()
        return top_items

## 使用範例

#model = AttentionRecommender(n_users, n_items, layers=1, device='cuda')
#model.fit(train_dt, epochs=5, lr=0.001)
#predictions = model.predict(test_dt)  # predictions 是每個使用者推薦的 item 陣列

class TransformerRecommender(nn.Module):
    def __init__(
        self,
        n_users,
        n_items,
        layers=2,
        device='cpu',
        d_model=64,
        max_seq_len=100
        ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.device = device
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embedding
        self.item_embedding = nn.Embedding(n_items, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Output head
        self.fc = nn.Linear(d_model, n_items)

        self.to(device)

    def forward(
        self,
        seq_tensor
        ):
        B, L = seq_tensor.shape

        # Embedding + Positional encoding
        item_emb = self.item_embedding(seq_tensor)
        positions = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        pos_emb = self.pos_embedding(positions)

        x = item_emb + pos_emb  # shape (B, L, D)
        x = self.encoder(x)     # Transformer encoding
        x = x[:, -1, :]         # 最後一個位置的輸出代表 summary
        logits = self.fc(x)     # 預測下一個 item 的分數
        return logits

    def fit(
        self,
        train_dt,
        epochs=10,
        lr=1e-3
        ):
        model = self.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        data = []
        # print("train_dt size:",len(train_dt))
        for uid, seq in train_dt.items():
            # print(f'uid:{uid} size is {len(seq)}')
            if len(seq) < 2:
                continue
            for i in range(len(seq)-5):
                input_seq = torch.tensor(seq[i:i+5], dtype=torch.long)
                target_item = seq[i+5]
                data.append((input_seq, target_item))
        # print("data size:",len(data))
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            total_loss = 0
            for seq, target in data:
                if len(seq) > self.max_seq_len:
                    seq = seq[-self.max_seq_len:]

                seq = seq.to(self.device).unsqueeze(0)
                target = torch.tensor([target], device=self.device)

                optimizer.zero_grad()
                logits = self(seq)
                loss = loss_fn(logits, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            tqdm.write(f"[Epoch {epoch + 1}/{epochs}] Total Loss: {total_loss:.4f}")
            # print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")

    def predict(
        self,
        test_dt,
        top_k=10
        ):
        self.eval()
        results = []
        with torch.no_grad():
            for uid, seq in test_dt.items():
                if len(seq) == 0:
                    results.append([])
                    continue

                if len(seq) > self.max_seq_len:
                    seq = seq[-self.max_seq_len:]
                seq_tensor = torch.tensor(seq, device=self.device).unsqueeze(0)
                logits = self(seq_tensor)
                top_items = torch.topk(logits, k=top_k, dim=1).indices.squeeze(0).tolist()
                results.append(top_items)
        return results
    
    def predict_one_seq(
        self,
        u_id,
        item2idx,
        seq,
        top_k=10
        ):
        self.eval()
        inv_map = {v: k for k, v in item2idx.items()}
        with torch.no_grad():
            if len(seq) == 0:
                return []
            if len(seq) > self.max_seq_len:
                seq = seq[-self.max_seq_len:]
        seq_tensor = torch.tensor(seq, device=self.device).unsqueeze(0)
        logits = self(seq_tensor)
        top_items = torch.topk(logits, k=top_k, dim=1).indices.squeeze(0).tolist()
        ranked = []
        for i in top_items:
            ranked.append(inv_map[i])
        
        return top_items

## 使用範例
#model = TransformerRecommender(n_users=n_users, n_items=n_items, device='cuda')
#model.fit(train_dt, epochs=5, lr=0.001)

# predict for a single user sequence
#user_seq = [12, 45, 78, 23]
#recommendations = model.predict_one_seq(user_seq, top_k=5)

