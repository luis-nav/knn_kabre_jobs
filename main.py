"""
Pipeline principal: preprocesamiento → kNN → evaluación → contrafactual de ejemplo.
"""

import os


from src.preprocessing import load_and_prepare
from src.knn import KNNClassifier
from src.evaluation import classification_report, print_report
from src.counterfactuals import reason_codes, generate_counterfactual, explain
from src.preprocessing import decode_vector

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")
K = 11   # cambia aquí después de la búsqueda de k óptimo


def main():
    # ------------------------------------------------------------------
    # 1. Cargar y preprocesar
    # ------------------------------------------------------------------
    print("Cargando y preprocesando datos...")
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare(
        DATA_PATH, test_ratio=0.2, seed=42
    )
    print(f"  Train: {len(X_train)} muestras  |  Test: {len(X_test)} muestras")
    n_pos = sum(y_train)
    print(
        f"  Train COMPLETED: {n_pos} ({n_pos/len(y_train):.1%})  |  otros: {len(y_train)-n_pos}")

    # ------------------------------------------------------------------
    # 2. Entrenar kNN
    # ------------------------------------------------------------------
    print(f"\nEntrenando KNNClassifier (k={K}, metric=euclidean)...")
    clf = KNNClassifier(k=K, metric="euclidean", weighted=False)
    clf.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 3. Evaluar en test set
    # ------------------------------------------------------------------
    print("Evaluando en test set...")
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    y_scores = [p.get(1, 0.0) for p in y_probas]

    report = classification_report(y_test, y_pred, y_scores)
    print_report(report)

    # ------------------------------------------------------------------
    # 4. Ejemplo de contrafactual: primer trabajo del test set clase 0
    # ------------------------------------------------------------------
    failed_indices = [i for i, label in enumerate(y_test) if label == 0]
    if not failed_indices:
        print("No hay trabajos no completados en el test set.")
        return

    example_idx = failed_indices[0]
    query = X_test[example_idx]

    decoded = decode_vector(list(query), feature_names, scaler)
    print(f"\nEjemplo de contrafactual para trabajo #{example_idx} (clase real: 0)")
    print("  Valores del trabajo:")
    for k, v in decoded.items():
        print(f"    {k:<14s} {v}")

    print("\nReason codes (features que más alejan de COMPLETED):")
    rc = reason_codes(query, clf, feature_names, target_class=1)
    for fname, score in rc[:8]:
        if fname.startswith("partition_"):
            real_val = decoded.get("Partition", "")
            display  = f"Partition={real_val}"
        elif fname.startswith("qos_"):
            real_val = decoded.get("QOS", "")
            display  = f"QOS={real_val}"
        else:
            real_val = decoded.get(fname, "")
            display  = fname
        print(f"  {display:<30s}  valor: {real_val:<20s}  contribución={score:.4f}")

    print("\nGenerando contrafactual...")
    result = generate_counterfactual(
        query, clf, feature_names, scaler, target_class=1
    )
    print(explain(result))


if __name__ == "__main__":
    main()
