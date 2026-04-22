from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_and_preprocess

RAW_FEATURES    = ['H2','CH4','C2H6','C2H4','C2H2','CO','CO2']
RATIO_FEATURES  = ['CH4_H2','C2H6_CH4','C2H2_C2H4','C2H4_C2H6','CO2_CO']
COMBINED        = RAW_FEATURES + RATIO_FEATURES

def ablation():
    df = load_and_preprocess()
    y  = df['FAULT_CLASS_LE']
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42)

    for label, feats in [("RAW only", RAW_FEATURES),
                          ("RATIO only", RATIO_FEATURES),
                          ("COMBINED", COMBINED)]:
        scores = cross_val_score(model, df[feats], y, cv=cv, scoring='f1_macro')
        print(f"{label:15s} | F1-macro: {scores.mean():.4f} ± {scores.std():.4f}")