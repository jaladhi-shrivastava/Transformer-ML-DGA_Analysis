import pandas as pd
from src.config import DATA_PATH, LABEL_MAP


def load_and_preprocess():
    df = pd.read_csv(DATA_PATH, sep=None, engine='python')

    # Normalize column names to uppercase, strip whitespace/special chars
    df.columns = (
        df.columns
        .str.strip()
        .str.upper()
        .str.replace(r'[^A-Z0-9_]', '', regex=True)
    )
    print("Columns:", df.columns.tolist())

    # Normalize label values
    df['FAULT_CLASS'] = (
        df['FAULT_CLASS']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Avoid division by zero — ONLY on gas columns, never on the label column
    gas_cols = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2']
    epsilon = 1e-6
    df[gas_cols] = df[gas_cols].replace(0, epsilon)

    # Feature engineering (ratio features)
    df['CH4_H2']    = df['CH4']  / df['H2']
    df['C2H6_CH4']  = df['C2H6'] / df['CH4']
    df['C2H2_C2H4'] = df['C2H2'] / df['C2H4']
    df['C2H4_C2H6'] = df['C2H4'] / df['C2H6']
    df['CO2_CO']    = df['CO2']  / df['CO']

    # Label encoding AFTER epsilon replacement so integers aren't corrupted
    df['FAULT_CLASS_LE'] = df['FAULT_CLASS'].map(LABEL_MAP)
    if df['FAULT_CLASS_LE'].isnull().any():
        print("Unmapped labels:", df[df['FAULT_CLASS_LE'].isnull()]['FAULT_CLASS'].unique())
        raise ValueError("Label mapping failed. Check LABEL_MAP in config.py.")
    df['FAULT_CLASS_LE'] = df['FAULT_CLASS_LE'].astype(int)

    return df