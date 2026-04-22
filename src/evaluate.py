import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from src.preprocessing import load_and_preprocess
from src.config import FEATURES, MODELS_DIR, LABEL_MAP_INV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def evaluate():
    df = load_and_preprocess()

    X = df[FEATURES]
    y = df['FAULT_CLASS_LE']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)

    target_names = [LABEL_MAP_INV[i] for i in sorted(LABEL_MAP_INV)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
    plt.show()
    print("Confusion matrix saved.")


if __name__ == "__main__":
    evaluate()