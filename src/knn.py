"""
kNN classifier implemented from scratch.

The algorithm logic (neighbour selection, voting, cross-validation) is fully
our own. numpy is used only for vectorised distance arithmetic — the same role
a BLAS library plays in any production implementation.
"""

from collections import Counter

import numpy as np


# --------------------------------------------------------------------------- #
# Distance matrices  (query_matrix × train_matrix → distance matrix)
# Each function accepts 2-D numpy arrays and returns a 1-D array of distances
# from a single query row (shape (d,)) to all training rows (shape (n, d)).
# --------------------------------------------------------------------------- #

def _euclidean_batch(query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """||query - x||₂  for every row x in X_train."""
    diff = X_train - query          # (n, d)
    return np.sqrt((diff * diff).sum(axis=1))


def _manhattan_batch(query: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Σ|query_i - x_i|  for every row x in X_train."""
    return np.abs(X_train - query).sum(axis=1)

# --------------------------------------------------------------------------- #
# KNN Classifier
# --------------------------------------------------------------------------- #


class KNNClassifier:
    """
    k-Nearest Neighbours classifier.

    Parameters
    ----------
    k : int
        Number of neighbours to use.
    metric : str
        Distance metric: "euclidean", "manhattan", or "minkowski".
    weighted : bool
        If True, neighbours vote weighted by 1/distance.
        If False, plain majority vote.
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weighted: bool = False,
    ):
        if metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError(f"Unknown metric '{metric}'.")
        self.k = k
        self.metric = metric
        self.weighted = weighted
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def fit(self, X, y) -> "KNNClassifier":
        """Store training data as numpy arrays (lazy learning)."""
        self._X = np.asarray(X, dtype=np.float64)
        self._y = np.asarray(y, dtype=np.int32)
        return self

    # ------------------------------------------------------------------ #
    def _distances(self, query: np.ndarray) -> np.ndarray:
        if self.metric == "euclidean":
            return _euclidean_batch(query, self._X)
        if self.metric == "manhattan":
            return _manhattan_batch(query, self._X)
        raise NotImplementedError("Metric not implemented yet.")

    def _get_neighbours(self, query: np.ndarray):
        """
        Returns (distances, labels) arrays for the k nearest neighbours,
        sorted ascending by distance.
        """
        dists = self._distances(query)
        # argpartition gives the k smallest indices (unsorted), then sort them
        k_idx = np.argpartition(dists, self.k)[: self.k]
        k_idx = k_idx[np.argsort(dists[k_idx])]
        return dists[k_idx], self._y[k_idx]

    # ------------------------------------------------------------------ #
    def predict_one(self, query) -> int:
        q = np.asarray(query, dtype=np.float64)
        dists, labels = self._get_neighbours(q)
        if self.weighted:
            weights = 1.0 / (dists + 1e-9)
            votes: dict[int, float] = {}
            for w, lbl in zip(weights, labels):
                votes[int(lbl)] = votes.get(int(lbl), 0.0) + float(w)
            return max(votes, key=votes.__getitem__)
        else:
            return int(Counter(labels.tolist()).most_common(1)[0][0])

    def predict(self, X) -> list[int]:
        X_arr = np.asarray(X, dtype=np.float64)
        return [self.predict_one(row) for row in X_arr]

    # ------------------------------------------------------------------ #
    def predict_proba_one(self, query) -> dict[int, float]:
        """Returns {class: probability} for the query point."""
        q = np.asarray(query, dtype=np.float64)
        dists, labels = self._get_neighbours(q)
        if self.weighted:
            weights = 1.0 / (dists + 1e-9)
            votes: dict[int, float] = {}
            for w, lbl in zip(weights, labels):
                votes[int(lbl)] = votes.get(int(lbl), 0.0) + float(w)
            total = sum(votes.values())
            return {cls: v / total for cls, v in votes.items()}
        else:
            total = len(labels)
            counts: dict[int, int] = {}
            for lbl in labels:
                counts[int(lbl)] = counts.get(int(lbl), 0) + 1
            return {cls: cnt / total for cls, cnt in counts.items()}

    def predict_proba(self, X) -> list[dict[int, float]]:
        X_arr = np.asarray(X, dtype=np.float64)
        return [self.predict_proba_one(row) for row in X_arr]

    # ------------------------------------------------------------------ #
    def get_neighbours_for(self, query) -> list[tuple[float, int]]:
        """Returns k nearest (distance, label) pairs sorted ascending."""
        q = np.asarray(query, dtype=np.float64)
        dists, labels = self._get_neighbours(q)
        return list(zip(dists.tolist(), labels.tolist()))


# --------------------------------------------------------------------------- #
# Cross-validation for k selection
# --------------------------------------------------------------------------- #

def cross_val_accuracy(
    X,
    y,
    k_values: list[int],
    n_folds: int = 5,
    metric: str = "euclidean",
) -> dict[int, float]:
    """
    Stratified k-fold cross-validation.
    Returns {k: mean_accuracy} for each k in k_values.
    """
    import random
    rng = random.Random(0)

    y_list = list(y) if not isinstance(y, list) else y

    idx_pos = [i for i, lbl in enumerate(y_list) if lbl == 1]
    idx_neg = [i for i, lbl in enumerate(y_list) if lbl == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def make_folds(indices):
        size = len(indices) // n_folds
        return [indices[i * size: (i + 1) * size] for i in range(n_folds)]

    folds_pos = make_folds(idx_pos)
    folds_neg = make_folds(idx_neg)
    folds = [folds_pos[i] + folds_neg[i] for i in range(n_folds)]

    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y_list, dtype=np.int32)

    results: dict[int, float] = {}

    for k in k_values:
        fold_accs = []
        for fold_i in range(n_folds):
            test_idx = folds[fold_i]
            train_idx = [idx for j, f in enumerate(
                folds) if j != fold_i for idx in f]

            clf = KNNClassifier(k=k, metric=metric)
            clf.fit(X_arr[train_idx], y_arr[train_idx])
            preds = clf.predict(X_arr[test_idx])
            acc = np.mean(np.array(preds) == y_arr[test_idx])
            fold_accs.append(float(acc))

        results[k] = sum(fold_accs) / len(fold_accs)
        print(f"  k={k:3d}  CV accuracy={results[k]:.4f}")

    return results
