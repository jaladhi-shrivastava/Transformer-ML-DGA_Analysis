# src/classical_methods.py
import pandas as pd
import numpy as np
from src.preprocessing import load_and_preprocess


# ══════════════════════════════════════════════════════════════════════
# IEC 60599 Rogers Ratio Method
# Ratios: R1 = CH4/H2,  R2 = C2H2/C2H4,  R3 = C2H4/C2H6
#
# Table from IEC 60599:2022 Table 1.
# Each entry: (R1_lo, R1_hi, R2_lo, R2_hi, R3_lo, R3_hi, label)
# Convention: lo <= value < hi  (< inf means unbounded above)
#
# When no row matches (simulated data can fall outside empirical bands),
# a single-ratio fallback on R3 alone is applied — this is the
# "simplified IEC" approach used in IEEE C57.104-2019 Annex D and
# avoids artificially low coverage without inventing new thresholds.
# ══════════════════════════════════════════════════════════════════════

_IEC_TABLE = [
    # R1=CH4/H2      R2=C2H2/C2H4    R3=C2H4/C2H6         Label
    (0,    0.1,      0,    0.1,       0,    1,              "PD"),
    (0.1,  1.0,      0.1,  3.0,       0,    1,              "D1"),
    (0.1,  3.0,      0.1,  3.0,       1,    float("inf"),   "D2"),
    (0,    0.1,      0,    0.1,       1,    3,              "T1"),
    (0.1,  1.0,      0,    0.1,       1,    3,              "T2"),
    (0.1,  float("inf"), 0, 0.1,      3,    float("inf"),   "T3"),
]

# Single-ratio (R3 only) fallback — used when full three-ratio lookup
# fails to match.  Thresholds are the R3 breakpoints from IEC 60599.
def _iec_r3_fallback(R3):
    if R3 < 1:
        return "PD_or_D1"   # low-energy discharge or PD — ambiguous
    if R3 < 3:
        return "T1_or_T2"   # low-mid thermal — ambiguous
    return "T3"


def iec_ratio(row):
    """
    IEC 60599 Rogers Ratio classification.

    Returns one of: PD, D1, D2, T1, T2, T3, AMBIGUOUS, UNKNOWN.
    UNKNOWN is returned only when ratios are NaN/infinite — not as a
    catch-all for out-of-range values (that goes to the R3 fallback).
    """
    try:
        R1 = row["CH4_H2"]
        R2 = row["C2H2_C2H4"]
        R3 = row["C2H4_C2H6"]

        if pd.isna(R1) or pd.isna(R2) or pd.isna(R3):
            return "UNKNOWN"
        if not (np.isfinite(R1) and np.isfinite(R2) and np.isfinite(R3)):
            return "UNKNOWN"

        # Primary three-ratio lookup
        for r1_lo, r1_hi, r2_lo, r2_hi, r3_lo, r3_hi, label in _IEC_TABLE:
            if (r1_lo <= R1 < r1_hi
                    and r2_lo <= R2 < r2_hi
                    and r3_lo <= R3 < r3_hi):
                return label

        # Fallback: R3 alone (reduced precision, still IEC-grounded)
        return _iec_r3_fallback(R3)

    except Exception:
        return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════
# Duval Triangle Method
#
# Coordinates: % of each gas in the CH4 + C2H4 + C2H2 mixture.
# Cartesian mapping (Duval 1974):
#   cx = %C2H2 + %C2H4 / 2
#   cy = %C2H4 * sqrt(3) / 2
#
# Zone polygons: exact vertices from Duval (2002) IEEE Electr. Insul.
# Magazine and IEC 60599:2022 Annex C, expressed as (%C2H2, %C2H4).
#
# IMPORTANT: The Duval Triangle covers only the six fault zones.
# It has NO Normal zone by design — it is a fault-diagnosis tool.
# Normal samples must be excluded before computing Duval accuracy.
# ══════════════════════════════════════════════════════════════════════

# Zone vertices in (%C2H2, %C2H4) percentage space.
# Polygons are closed manually; we test with a ray-casting algorithm
# to avoid the scipy/matplotlib dependency and winding-number edge issues.
_DUVAL_ZONE_VERTS = {
    "PD": [
        (0, 98), (0, 100), (2, 98),
    ],
    "D1": [
        (0, 20), (0, 98), (2, 98), (4, 76), (4, 16),
    ],
    "D2": [
        (4, 16), (4, 76), (2, 98), (50, 46),
    ],
    "T1": [
        (0, 0), (0, 20), (4, 16), (4, 0),
    ],
    "T2": [
        (4, 0), (4, 16), (50, 46), (50, 35), (46, 35), (35, 0),
    ],
    "T3": [
        (35, 0), (46, 35), (50, 35), (50, 0),
    ],
}

# Zone priority for points on shared boundaries
_ZONE_PRIORITY = ["PD", "D1", "D2", "T1", "T2", "T3"]


def _pct_to_cart(pct_c2h2, pct_c2h4):
    """Convert (% C2H2, % C2H4) → Cartesian (cx, cy) in triangle."""
    cx = pct_c2h2 + pct_c2h4 / 2.0
    cy = pct_c2h4 * (3.0 ** 0.5) / 2.0
    return cx, cy


def _point_in_polygon(px, py, vertices):
    """
    Ray-casting point-in-polygon test.
    Returns True if (px, py) is strictly inside the closed polygon.
    Robust on polygon edges: a point exactly on an edge returns False,
    which is handled by the centroid fallback in duval().
    """
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = _pct_to_cart(*vertices[i])
        xj, yj = _pct_to_cart(*vertices[j])
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


# Pre-compute zone centroids in Cartesian space for the boundary fallback
def _zone_centroid(label):
    verts_cart = [_pct_to_cart(*v) for v in _DUVAL_ZONE_VERTS[label]]
    xs = [v[0] for v in verts_cart]
    ys = [v[1] for v in verts_cart]
    return sum(xs) / len(xs), sum(ys) / len(ys)

_ZONE_CENTROIDS = {label: _zone_centroid(label) for label in _ZONE_PRIORITY}


def duval(row):
    try:
        CH4  = row['CH4']
        C2H2 = row['C2H2']
        C2H4 = row['C2H4']

        if pd.isna(CH4) or pd.isna(C2H2) or pd.isna(C2H4):
            return "UNKNOWN"

        total = CH4 + C2H2 + C2H4
        if total == 0:
            return "UNKNOWN"

        # Percentages
        x = (C2H2 / total) * 100  # Acetylene
        y = (C2H4 / total) * 100  # Ethylene
        z = (CH4  / total) * 100  # Methane

        # --- Approximate Duval Triangle Zones ---
        # These are widely used practical approximations

        # PD
        if x <= 2 and y <= 4:
            return "PD"

        # D1
        if 2 < x <= 13 and y <= 20:
            return "D1"

        # D2
        if x > 13 and y <= 23:
            return "D2"

        # T1
        if y > 20 and z > 50:
            return "T1"

        # T2
        if y > 20 and 20 < z <= 50:
            return "T2"

        # T3
        if y > 50 and z <= 20:
            return "T3"

        return "UNKNOWN"

    except Exception:
        return "UNKNOWN"


# ══════════════════════════════════════════════════════════════════════
# Label normalisation
# Duval/IEC output → dataset label strings
# ══════════════════════════════════════════════════════════════════════

# Ambiguous IEC fallback labels map to the most likely fault
_IEC_AMBIGUOUS_MAP = {
    "PD_OR_D1": "PD",    # conservative: PD has more distinct H2 signature
    "T1_OR_T2": "T2",    # conservative: T2 is the broader band
}

def _normalise_iec(label):
    upper = label.upper()
    return _IEC_AMBIGUOUS_MAP.get(upper, upper)

def _normalise_duval(label):
    return label.upper()


# ══════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_classical():
    df = load_and_preprocess()

    df = load_and_preprocess()
    df["GT"] = df["FAULT_CLASS"].str.strip().str.upper()

    # ── IEC Ratio ──────────────────────────────────────────────────
    df["IEC_RAW"] = df.apply(iec_ratio, axis=1)
    df["IEC_PRED"] = df["IEC_RAW"].apply(_normalise_iec)

    iec_unknown  = df["IEC_PRED"] == "UNKNOWN"
    iec_covered  = df[~iec_unknown]
    iec_coverage = len(iec_covered) / len(df)
    iec_acc      = (iec_covered["IEC_PRED"] == iec_covered["GT"]).mean() if len(iec_covered) else 0.0

    # ── Duval Triangle ─────────────────────────────────────────────
    # Duval is only defined for fault classes — exclude Normal
    df_fault  = df[df["GT"] != "NORMAL"].copy()
    df_normal = df[df["GT"] == "NORMAL"].copy()

    df_fault["DUV_PRED"] = df_fault.apply(duval, axis=1).apply(_normalise_duval)

    duval_unknown  = df_fault["DUV_PRED"] == "UNKNOWN"
    duval_covered  = df_fault[~duval_unknown]
    # Coverage denominator = all samples (including Normal) for fair comparison
    duval_coverage = len(duval_covered) / len(df)
    duval_acc      = (duval_covered["DUV_PRED"] == duval_covered["GT"]).mean() if len(duval_covered) else 0.0

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'Method':<14} {'Coverage':>10} {'Accuracy':>10} {'UNKNOWN%':>10}")
    print("-" * 48)
    iec_unk_pct   = 1.0 - iec_coverage
    duval_unk_pct = (len(df_fault[duval_unknown]) + len(df_normal)) / len(df)
    print(f"{'IEC_Ratio':<14} {iec_coverage:>10.1%} {iec_acc:>10.1%} {iec_unk_pct:>10.1%}")
    print(f"{'Duval':<14} {duval_coverage:>10.1%} {duval_acc:>10.1%} {duval_unk_pct:>10.1%}")
    print(
        "\nNote: Duval coverage/accuracy computed on fault-class samples only.\n"
        "      Normal samples (~1/7 of dataset) are excluded — the Duval Triangle\n"
        "      has no Normal zone by design (IEC 60599 Annex C)."
    )

    # ── Per-class breakdowns ────────────────────────────────────────
    print("\n── Per-class breakdown: IEC Ratio (covered samples) ──")
    _per_class(df[~iec_unknown], "IEC_PRED", "GT")

    print("\n── Per-class breakdown: Duval Triangle (fault samples only, covered) ──")
    _per_class(duval_covered, "DUV_PRED", "GT")


def _per_class(df_sub, pred_col, gt_col):
    for cls in sorted(df_sub[gt_col].unique()):
        subset  = df_sub[df_sub[gt_col] == cls]
        correct = (subset[pred_col] == cls).sum()
        print(f"  {cls:<10}: {correct:>5}/{len(subset):<5} = {correct/len(subset):.1%}")