"""
Contrafactuales y reason codes para el clasificador kNN.

Dado un trabajo clasificado como NO completado (clase 0), este módulo:
  1. Calcula reason codes: qué features alejan al trabajo de la clase COMPLETED.
  2. Genera un contrafactual: la mínima perturbación de features modificables
     que haría que el clasificador lo predijera como COMPLETED.
"""

import numpy as np
from src.knn import KNNClassifier
from src.preprocessing import decode_vector,

from globals import PARTITIONS, QOS_VALUES, MUTABLE_FEATURES


def _feature_index(name: str, feature_names: list[str]) -> int | None:
    try:
        return feature_names.index(name)
    except ValueError:
        return None


def _mutable_indices(feature_names: list[str]) -> list[int]:
    """Return indices of all mutable features (including partition/QOS one-hots)."""
    indices = []
    for i, name in enumerate(feature_names):
        for mutable in MUTABLE_FEATURES:
            if name == mutable or name.startswith("partition_") or name.startswith("qos_"):
                indices.append(i)
                break
    return list(dict.fromkeys(indices))   # deduplicate, preserve order


# --------------------------------------------------------------------------- #
# Reason codes
# --------------------------------------------------------------------------- #

def reason_codes(
    query: list[float],
    clf: KNNClassifier,
    feature_names: list[str],
    target_class: int = 1,
) -> list[tuple[str, float]]:
    """
    For each feature, compute how much it contributes to the distance
    between the query and the nearest neighbours of the target class.

    Returns a list of (feature_name, contribution) sorted descending —
    the first entries are the features that push the query furthest from
    the target class.
    """
    # Get k nearest neighbours overall
    neighbours = clf.get_neighbours_for(query)

    # Filter only those belonging to the target class
    target_neighbours = [(dist, idx) for dist, label in neighbours if label == target_class
                         for idx in [None]]  # we need the actual vectors

    # Filter training points that belong to the target class
    mask = clf._y == target_class
    target_vecs = clf._X[mask]

    if len(target_vecs) == 0:
        return []

    # Find k nearest target-class points (vectorised)
    q = np.asarray(query, dtype=np.float64)
    dists = np.sqrt(((target_vecs - q) ** 2).sum(axis=1))
    k_idx = np.argpartition(dists, min(clf.k, len(dists) - 1))[: clf.k]
    nearest = target_vecs[k_idx]   # shape (k, d)

    # Per-feature contribution = mean absolute difference across nearest neighbours
    mean_diffs = np.abs(nearest - q).mean(axis=0)
    contributions = list(zip(feature_names, mean_diffs.tolist()))

    contributions.sort(key=lambda t: -t[1])
    return contributions


# --------------------------------------------------------------------------- #
# Contrafactual generator — greedy feature perturbation
# --------------------------------------------------------------------------- #

def generate_counterfactual(
    query: list[float],
    clf: KNNClassifier,
    feature_names: list[str],
    scaler,                        # fitted MinMaxScaler (for bounds)
    target_class: int = 1,
    step_size: float = 0.05,
    max_iterations: int = 500,
) -> dict:
    """
    Greedy search for a counterfactual: at each step move along the mutable
    feature that most increases P(target_class), until the classifier flips.

    Strategy: try steps of decreasing size (0.2, 0.1, 0.05) so we can take
    large jumps early and fine-tune near the boundary.

    Returns a dict with:
        "found"         : bool — whether we crossed the decision boundary
        "counterfactual": list[float] — the perturbed feature vector (scaled)
        "changes"       : list[dict]  — per-feature changes
        "proba_original": float — P(target_class) for original query
        "proba_cf"      : float — P(target_class) for counterfactual
    """
    mut_idx = _mutable_indices(feature_names)

    # Sort mutable features by reason-code contribution
    rc = reason_codes(query, clf, feature_names, target_class)
    rc_map = {name: score for name, score in rc}
    mut_idx_sorted = sorted(
        mut_idx,
        key=lambda i: -rc_map.get(feature_names[i], 0.0),
    )

    current = list(query)
    original_proba = clf.predict_proba_one(query).get(target_class, 0.0)
    step_sizes = [0.20, 0.10, 0.05, 0.02]

    for iteration in range(max_iterations):
        if clf.predict_one(current) == target_class:
            break

        best_gain = -1.0
        best_candidate = None
        current_proba = clf.predict_proba_one(current).get(target_class, 0.0)

        for j in mut_idx_sorted:
            for s in step_sizes:
                for direction in (+s, -s):
                    candidate = list(current)
                    candidate[j] = max(0.0, min(1.0, candidate[j] + direction))
                    proba = clf.predict_proba_one(
                        candidate).get(target_class, 0.0)
                    gain = proba - current_proba
                    if gain > best_gain:
                        best_gain = gain
                        best_candidate = candidate

        if best_candidate is None or best_gain <= 1e-6:
            break
        current = best_candidate

    found = clf.predict_one(current) == target_class
    cf_proba = clf.predict_proba_one(current).get(target_class, 0.0)

    # Decode both vectors to human-readable values
    orig_decoded = decode_vector(list(query), feature_names, scaler)
    cf_decoded = decode_vector(current,     feature_names, scaler)

    # Collect changes — for one-hot groups report at the group level
    ONE_HOT_GROUPS = {
        "Partition": [n for n in feature_names if n.startswith("partition_")],
        "QOS":       [n for n in feature_names if n.startswith("qos_")],
    }
    one_hot_feature_names = {n for g in ONE_HOT_GROUPS.values() for n in g}

    changes = []
    reported_groups: set[str] = set()

    for j, fname in enumerate(feature_names):
        delta = current[j] - query[j]
        if abs(delta) < 1e-6:
            continue

        # Skip raw one-hot columns — handled as a group below
        if fname in one_hot_feature_names:
            for group_name, members in ONE_HOT_GROUPS.items():
                if fname in members and group_name not in reported_groups:
                    orig_val = orig_decoded.get(group_name, "?")
                    cf_val = cf_decoded.get(group_name, "?")
                    if orig_val != cf_val:
                        changes.append({
                            "feature":       group_name,
                            "original_real": orig_val,
                            "cf_real":       cf_val,
                            "delta_scaled":  None,   # no single delta for groups
                        })
                        reported_groups.add(group_name)
            continue

        # Numeric feature
        orig_val = orig_decoded.get(fname, f"{query[j]:.3f}")
        cf_val = cf_decoded.get(fname,   f"{current[j]:.3f}")
        changes.append({
            "feature":       fname,
            "original_real": orig_val,
            "cf_real":       cf_val,
            "delta_scaled":  delta,
        })

    return {
        "found":          found,
        "counterfactual": current,
        "changes":        changes,
        "proba_original": original_proba,
        "proba_cf":       cf_proba,
        "original_decoded": orig_decoded,
        "cf_decoded":       cf_decoded,
    }


# --------------------------------------------------------------------------- #
# Human-readable explanation
# --------------------------------------------------------------------------- #

def explain(result: dict) -> str:
    lines = []
    status = "ENCONTRADO" if result["found"] else "NO encontrado (aproximado)"
    lines.append(f"Contrafactual: {status}")
    lines.append(
        f"P(COMPLETED): {result['proba_original']:.2%} → {result['proba_cf']:.2%}"
    )

    if result["changes"]:
        lines.append("\nCambios sugeridos:")
        for ch in result["changes"]:
            delta = ch["delta_scaled"]
            if delta is None:
                arrow = "↔"   # categorical change
            elif delta > 0:
                arrow = "▲"
            else:
                arrow = "▼"
            lines.append(
                f"  {arrow} {ch['feature']:<14s}  "
                f"{ch['original_real']}  →  {ch['cf_real']}"
            )
    else:
        lines.append("Sin cambios necesarios (ya clasificado como COMPLETED).")

    return "\n".join(lines)
