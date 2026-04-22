from src.train import train_model
from src.benchmark import benchmark_all
from src.feature_ablation import ablation
from src.classical_methods import evaluate_classical
from src.explain import explain_model


def main():
    print("\n==============================")
    print(" STEP 4: CLASSICAL METHODS")
    print("==============================")
    evaluate_classical()

    print("\n==============================")
    print(" STEP 1: TRAINING MODELS")
    print("==============================")
    train_model()

    print("\n==============================")
    print(" STEP 2: MODEL BENCHMARKING")
    print("==============================")
    results, comparison_df = benchmark_all()

    print("\n==============================")
    print(" STEP 3: FEATURE ABLATION")
    print("==============================")
    ablation()

    print("\n==============================")
    print(" STEP 5: EXPLAINABILITY (SHAP)")
    print("==============================")
    explain_model()

    print("\n==============================")
    print(" ALL TASKS COMPLETED")
    print("==============================")


if __name__ == "__main__":
    main()