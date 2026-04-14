import csv
from datetime import datetime

from .globals import PARTITIONS, QOS_VALUES


# --------------------------------------------------------------------------- #
# ReqMem parser
# Slurm format: <value><unit>  where unit in {Mc, Mn, Gc, Gn}
#   M = megabytes,  G = gigabytes
#   c = per core,   n = per node
# We convert everything to MB (ignoring per-core/per-node since ReqCPUS
# and ReqNodes are separate features).
# --------------------------------------------------------------------------- #

def parse_reqmem_mb(raw: str) -> float:
    raw = raw.strip()
    if raw.endswith(("Mc", "Mn")):
        return float(raw[:-2])
    if raw.endswith(("Gc", "Gn")):
        return float(raw[:-2]) * 1024.0
    # fallback: try plain number
    return float(raw)


def parse_submit(raw: str):
    """Return (hour_of_day, day_of_week) from ISO-8601 datetime string."""
    dt = datetime.fromisoformat(raw)
    return dt.hour, dt.weekday()   # weekday: 0=Mon … 6=Sun


def normalize_state(raw: str) -> int:
    """1 = COMPLETED, 0 = everything else."""
    return 1 if raw.strip() == "COMPLETED" else 0


# --------------------------------------------------------------------------- #
# Load raw CSV
# --------------------------------------------------------------------------- #

def load_raw(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        for row in reader:
            rows.append(row)
    return rows


def one_hot(value: str, categories: list[str]) -> list[float]:
    return [1.0 if value.strip() == c else 0.0 for c in categories]


def build_features(row: dict) -> list[float] | None:
    """
    Returns a feature vector or None if the row has critical missing values.

    Feature vector layout (27 dims):
      [0]  ReqCPUS
      [1]  ReqMem_MB
      [2]  ReqNodes
      [3]  ResvCPURAW
      [4]  TimelimitRaw
      [5]  Priority
      [6]  submit_hour
      [7]  submit_dow
      [8..31]  one-hot Partition  (24 categories)
      [32..33] one-hot QOS        (2 categories)
    """
    try:
        req_cpus = float(row["ReqCPUS"])
        req_mem = parse_reqmem_mb(row["ReqMem"])
        req_nodes = float(row["ReqNodes"])
        resv_cpu = float(row["ResvCPURAW"])
        timelimit = float(row["TimelimitRaw"])
        priority = float(row["Priority"])
        hour, dow = parse_submit(row["Submit"])
    except (ValueError, KeyError):
        return None

    vec = [
        req_cpus,
        req_mem,
        req_nodes,
        resv_cpu,
        timelimit,
        priority,
        float(hour),
        float(dow),
        *one_hot(row["Partition"], PARTITIONS),
        *one_hot(row["QOS"],       QOS_VALUES),
    ]
    return vec


# --------------------------------------------------------------------------- #
# Min-max scaler (fitted on training data only)
# --------------------------------------------------------------------------- #

class MinMaxScaler:
    def __init__(self):
        self.mins: list[float] = []
        self.maxs: list[float] = []

    def fit(self, X: list[list[float]]) -> "MinMaxScaler":
        n_feat = len(X[0])
        self.mins = [min(row[j] for row in X) for j in range(n_feat)]
        self.maxs = [max(row[j] for row in X) for j in range(n_feat)]
        return self

    def transform(self, X: list[list[float]]) -> list[list[float]]:
        scaled = []
        for row in X:
            new_row = []
            for j, val in enumerate(row):
                denom = self.maxs[j] - self.mins[j]
                new_row.append(
                    (val - self.mins[j]) / denom if denom > 0 else 0.0)
            scaled.append(new_row)
        return scaled

    def inverse_transform_one(self, row: list[float]) -> list[float]:
        """Recover original-scale values from a single scaled row."""
        return [
            val * (self.maxs[j] - self.mins[j]) + self.mins[j]
            for j, val in enumerate(row)
        ]

    def fit_transform(self, X: list[list[float]]) -> list[list[float]]:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# Stratified train/test split (no external libraries)
# --------------------------------------------------------------------------- #

def split_list(lst, test_ratio: float = 0.2) -> tuple:
    cut = int(len(lst) * test_ratio)
    return lst[cut:], lst[:cut]   # train, test


def stratified_split(
    X: list[list[float]],
    y: list[int],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple:
    """Returns X_train, X_test, y_train, y_test."""
    import random
    rng = random.Random(seed)

    idx_pos = [i for i, label in enumerate(y) if label == 1]
    idx_neg = [i for i, label in enumerate(y) if label == 0]

    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    train_pos, test_pos = split_list(idx_pos)
    train_neg, test_neg = split_list(idx_neg)

    train_idx = train_pos + train_neg
    test_idx = test_pos + test_neg
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

def load_and_prepare(path: str, test_ratio: float = 0.2, seed: int = 42):
    """
    Loads the CSV, builds features, scales, and splits.

    Returns:
        X_train, X_test, y_train, y_test  (all plain Python lists)
        scaler                             (fitted MinMaxScaler)
        feature_names                      (list of str)
    """
    rows = load_raw(path)

    X_raw, y = [], []
    for row in rows:
        vec = build_features(row)
        if vec is None:
            continue
        label = normalize_state(row["State"])
        X_raw.append(vec)
        y.append(label)

    X_train_raw, X_test_raw, y_train, y_test = stratified_split(
        X_raw, y, test_ratio=test_ratio, seed=seed
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    feature_names = [
        "ReqCPUS", "ReqMem_MB", "ReqNodes", "ResvCPURAW",
        "TimelimitRaw", "Priority", "submit_hour", "submit_dow",
        *[f"partition_{p}" for p in PARTITIONS],
        *[f"qos_{q}" for q in QOS_VALUES],
    ]

    return X_train, X_test, y_train, y_test, scaler, feature_names


# --------------------------------------------------------------------------- #
# Human-readable decoding of a (scaled) feature vector
# --------------------------------------------------------------------------- #

_DOW_NAMES = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


def _fmt_memory(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb:.0f} MB"


def _fmt_timelimit(minutes: float) -> str:
    minutes = int(round(minutes))
    if minutes >= 1440:
        d = minutes // 1440
        h = (minutes % 1440) // 60
        return f"{d}d {h}h" if h else f"{d}d"
    if minutes >= 60:
        h = minutes // 60
        m = minutes % 60
        return f"{h}h {m}m" if m else f"{h}h"
    return f"{minutes}m"


def decode_vector(
    scaled_row: list[float],
    feature_names: list[str],
    scaler: "MinMaxScaler",
) -> dict:
    """
    Convert a scaled feature vector to a human-readable dict.

    Numeric features are inverse-transformed and shown with units.
    One-hot groups (partition_*, qos_*) are decoded to the active category name.

    Returns: {display_name: formatted_string}
    """
    raw = scaler.inverse_transform_one(scaled_row)

    result = {}

    # --- Numeric features ------------------------------------------------- #
    numeric_fmt = {
        "ReqCPUS": lambda v: f"{int(round(v))} CPUs",
        "ReqMem_MB": lambda v: _fmt_memory(v),
        "ReqNodes": lambda v: f"{int(round(v))} nodos",
        "ResvCPURAW": lambda v: f"{int(round(v))} CPU-s reservados",
        "TimelimitRaw": lambda v: _fmt_timelimit(v),
        "Priority": lambda v: f"{int(round(v))}",
        "submit_hour": lambda v: f"{int(round(v)):02d}:00 h",
        "submit_dow": lambda v: _DOW_NAMES[min(6, max(0, int(round(v))))],
    }
    for name, fmt in numeric_fmt.items():
        if name in feature_names:
            idx = feature_names.index(name)
            result[name] = fmt(raw[idx])

    # --- One-hot: Partition ------------------------------------------------ #
    part_indices = [i for i, n in enumerate(
        feature_names) if n.startswith("partition_")]
    if part_indices:
        best_i = max(part_indices, key=lambda i: scaled_row[i])
        result["Partition"] = feature_names[best_i].replace("partition_", "")

    # --- One-hot: QOS ------------------------------------------------------ #
    qos_indices = [i for i, n in enumerate(
        feature_names) if n.startswith("qos_")]
    if qos_indices:
        best_i = max(qos_indices, key=lambda i: scaled_row[i])
        result["QOS"] = feature_names[best_i].replace("qos_", "")

    return result
