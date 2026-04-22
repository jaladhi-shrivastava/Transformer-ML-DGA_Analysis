import os
import joblib
import pandas as pd
from src.config import FEATURES, MODELS_DIR, LABEL_MAP_INV


def load_model():
    model_path = os.path.join(MODELS_DIR, "random_forest.pkl")
    return joblib.load(model_path)


def predict_fault(data_dict):
    """
    data_dict: dict with keys matching raw gas columns (H2, CH4, C2H6, C2H4, C2H2, CO, CO2)
    Returns: predicted fault label string (e.g. 'D1', 'T2', 'NORMAL')
    """
    epsilon = 1e-6
    d = {k: (v if v != 0 else epsilon) for k, v in data_dict.items()}

    # Compute ratio features
    d['CH4_H2']    = d['CH4']  / d['H2']
    d['C2H6_CH4']  = d['C2H6'] / d['CH4']
    d['C2H2_C2H4'] = d['C2H2'] / d['C2H4']
    d['C2H4_C2H6'] = d['C2H4'] / d['C2H6']
    d['CO2_CO']    = d['CO2']  / d['CO']

    df = pd.DataFrame([d])[FEATURES]

    model = load_model()
    prediction_int = model.predict(df)[0]
    return LABEL_MAP_INV[prediction_int]