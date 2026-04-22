import os, joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from src.preprocessing import load_and_preprocess
from src.config import FEATURES, MODELS_DIR, LABEL_MAP_INV

MODELS = {
    "random_forest": "random_forest.pkl",
    "decision_tree": "decision_tree.pkl",
    "xgboost":       "xgboost.pkl"
}

def benchmark_all():
    df = load_and_preprocess()
    X = df[FEATURES]
    y = df['FAULT_CLASS_LE']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV)]
    results = {}

    for name, fname in MODELS.items():
        path = os.path.join(MODELS_DIR, fname)

        if not os.path.exists(path):
            print(f"Skipping {name}: model not found")
            continue

        model = joblib.load(path)
        y_pred = model.predict(X_test)

        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True
        )

        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "report": report,
            "confusion_matrix": cm.tolist()
        }

        print(f"\n=== {name.upper()} ===")
        print(classification_report(y_test, y_pred, target_names=target_names))

    # Fault-wise comparison
    print("\n=== FAULT-WISE F1 COMPARISON ===")

    rows = []
    for name, res in results.items():
        for fault in target_names:
            rows.append({
                "model": name,
                "fault": fault,
                "f1": res["report"][fault]["f1-score"],
                "precision": res["report"][fault]["precision"],
                "recall": res["report"][fault]["recall"]
            })

    comparison_df = pd.DataFrame(rows)

    pivot = comparison_df.pivot(index="fault", columns="model", values="f1")
    print(pivot.round(3))

    pivot["best_model"] = pivot.idxmax(axis=1)

    print("\nBest model per fault class:")
    print(pivot["best_model"])


    save_path = os.path.join(MODELS_DIR, "model_comparison.csv")
    comparison_df.to_csv(save_path, index=False)
    print(f"\nSaved comparison to: {save_path}")

    return results, comparison_df