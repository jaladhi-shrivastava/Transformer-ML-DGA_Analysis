import numpy as np
import shap, joblib, os
import matplotlib.pyplot as plt
from src.preprocessing import load_and_preprocess
from src.config import FEATURES, MODELS_DIR, LABEL_MAP_INV

def explain_model(n_samples=200):

    print("Generating SHAP explanations...")

    df = load_and_preprocess()
    X  = df[FEATURES].sample(n_samples, random_state=42)

    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")

    if not os.path.exists(model_path):
        print("Model not found. Train model first.")
        return

    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    print("SHAP shape:", shap_values.shape)

    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        # FIXED AXIS
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_list = shap_values
    else:
        shap_list = [shap_values]

    print(f"Detected {len(shap_list)} classes")

    for i, shap_val in enumerate(shap_list):
        label = LABEL_MAP_INV.get(i, f"class_{i}")

        plt.figure()

        shap.summary_plot(
            shap_val,
            X,
            feature_names=FEATURES,
            show=False,
            plot_type="bar"
        )

        plt.title(f"SHAP Feature Importance — {label}")
        plt.tight_layout()

        save_path = os.path.join(MODELS_DIR, f"shap_{label}.png")
        plt.savefig(save_path)
        plt.close()

        print(f"Saved SHAP plot: {save_path}")