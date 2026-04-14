"""
Evaluation metrics — implemented from scratch.
"""


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #

def confusion_matrix(y_true: list[int], y_pred: list[int]) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def accuracy(y_true, y_pred) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def precision(cm: dict) -> float:
    denom = cm["TP"] + cm["FP"]
    return cm["TP"] / denom if denom > 0 else 0.0


def recall(cm: dict) -> float:
    denom = cm["TP"] + cm["FN"]
    return cm["TP"] / denom if denom > 0 else 0.0


def f1_score(cm: dict) -> float:
    p, r = precision(cm), recall(cm)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_auc(y_true: list[int], y_scores: list[float]) -> float:
    """
    Compute AUC-ROC using the trapezoidal rule.
    y_scores: probability of the positive class (class=1).
    """
    # Sort by descending score
    pairs = sorted(zip(y_scores, y_true), key=lambda t: -t[0])

    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    tpr_list, fpr_list = [0.0], [0.0]
    tp = fp = 0

    for _, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * \
            (tpr_list[i] + tpr_list[i - 1]) / 2
    return auc


# --------------------------------------------------------------------------- #
# Pretty report
# --------------------------------------------------------------------------- #

def classification_report(
    y_true: list[int],
    y_pred: list[int],
    y_scores: list[float] | None = None,
) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy(y_true, y_pred)
    prec = precision(cm)
    rec = recall(cm)
    f1 = f1_score(cm)

    report = {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "confusion_matrix": cm,
    }

    if y_scores is not None:
        report["auc_roc"] = roc_auc(y_true, y_scores)

    return report


def print_report(report: dict) -> None:
    cm = report["confusion_matrix"]
    print(f"\n{'='*40}")
    print(f"  Accuracy : {report['accuracy']:.4f}")
    print(f"  Precision: {report['precision']:.4f}")
    print(f"  Recall   : {report['recall']:.4f}")
    print(f"  F1       : {report['f1']:.4f}")
    if "auc_roc" in report:
        print(f"  AUC-ROC  : {report['auc_roc']:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0    {cm['TN']:6d}   {cm['FP']:6d}")
    print(f"  Actual 1    {cm['FN']:6d}   {cm['TP']:6d}")
    print(f"{'='*40}\n")
