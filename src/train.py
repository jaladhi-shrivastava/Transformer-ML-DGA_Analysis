import os
import joblib
import numpy as np

from src.preprocessing import load_and_preprocess
from src.config import FEATURES, MODELS_DIR

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def train_model():

    # =========================
    # LOAD DATA
    # =========================
    df = load_and_preprocess()

    X = df[FEATURES]
    y = df['FAULT_CLASS_LE']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # =========================
    # MODELS DEFINITION
    # =========================
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=42
        )
    }

    # =========================
    # CROSS VALIDATION SETUP
    # =========================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    os.makedirs(MODELS_DIR, exist_ok=True)

    # =========================
    # TRAIN + VALIDATE EACH MODEL
    # =========================
    for name, model in models.items():

        print(f"\n=== Training {name.upper()} ===")

        # Cross-validation
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring='f1_macro'
        )

        print(f"CV F1 (macro): {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # Train on full training set
        model.fit(X_train, y_train)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

        print(f"{name} saved at: {model_path}")


if __name__ == "__main__":
    train_model()