



from joblib import Parallel, delayed
from typing import List, Union, Sequence
import numpy as np
class PlackettLuce:
    def __init__(
        self,
        max_iter: int = 1000,
        tol: float = 1e-8,
        n_jobs: int = None
    ):
        """
        Plackett–Luce ranking model via MM.
        
        Args:
          max_iter: maximum MM iterations
          tol: convergence threshold on max parameter change
          n_jobs: number of parallel workers for batch predict (None or 1 = no parallelism)
        """
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        self.id2idx = {}
        self.idx2id = {}
        self.theta = None
        self.fitted = False

    def fit(
        self,
        rankings: List[List[int]],
        unique_ids: Sequence[int]
    ) -> None:
        """
        Estimate worth parameters from observed rankings.

        Args:
          rankings: list of observed ranked lists (item-IDs)
          unique_ids: list of all possible item-IDs
        """
        # build ID ↔ index maps
        self.id2idx = {item: idx for idx, item in enumerate(unique_ids)}
        self.idx2id = {idx: item for item, idx in self.id2idx.items()}
        n = len(unique_ids)

        # translate rankings to index space
        idx_rankings = [
            [self.id2idx[i] for i in r if i in self.id2idx]
            for r in rankings
        ]

        # initialize uniform worth
        theta = np.ones(n, dtype=float) / n

        # MM iterations
        for it in range(self.max_iter):
            theta_new = np.zeros_like(theta)
            for r in idx_rankings:
                if not r:
                    continue
                denom = theta[r].sum()
                for j in r:
                    theta_new[j] += 1.0 / denom
                    denom -= theta[j]
            theta_new /= theta_new.sum()

            if np.max(np.abs(theta_new - theta)) < self.tol:
                theta = theta_new
                break
            theta = theta_new
        else:
            print("Warning: PlackettLuce.fit did not converge within max_iter")

        self.theta = theta
        self.fitted = True

    def _predict_one(self, seq: Sequence[int]) -> List[int]:
        """Sort a single sequence by descending worth."""
        return sorted(
            seq,
            key=lambda x: -self.theta[self.id2idx[x]]
                     if (self.fitted and x in self.id2idx) else 0.0
        )

    def predict(
        self,
        seqs: Union[Sequence[int], List[Sequence[int]]]
    ) -> Union[List[int], List[List[int]]]:
        """
        Predict ordering(s) by descending worth.

        Args:
          seqs: either
            - a single Sequence[int], or
            - a List of Sequence[int]

        Returns:
          - for a single sequence: List[int]
          - for multiple: List[List[int]]
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before calling predict()")

        # detect batch vs single
        is_batch = (
            isinstance(seqs, list) and
            (not seqs or isinstance(seqs[0], Sequence))
        )

        if not is_batch:
            # single sequence
            return self._predict_one(seqs)  # type: ignore

        # batch of sequences
        if self.n_jobs and self.n_jobs != 1:
            # parallel
            return Parallel(n_jobs=self.n_jobs)(
                delayed(self._predict_one)(seq) for seq in seqs
            )
        else:
            # serial
            return [self._predict_one(seq) for seq in seqs]

    def get_worth(self) -> dict[int, float]:
        """
        Returns:
          mapping from item-ID → its estimated worth
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")
        return { self.idx2id[i]: self.theta[i] for i in range(len(self.theta)) }


