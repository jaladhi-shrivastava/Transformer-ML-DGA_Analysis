import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transformer_dga_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Must match uppercased column names from preprocessing
FEATURES = [
    'H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2',
    'CH4_H2', 'C2H6_CH4', 'C2H2_C2H4', 'C2H4_C2H6', 'CO2_CO'
]

LABEL_MAP = {
    'PD': 0,
    'D1': 1,
    'D2': 2,
    'T1': 3,
    'T2': 4,
    'T3': 5,
    'NORMAL': 6
}

# Inverse map: int -> label string
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}