# Transformer Fault Detection using Machine Learning (DGA-Based)

## Overview

This project implements a machine learning-based system for detecting transformer faults using Dissolved Gas Analysis (DGA).  
It focuses on identifying incipient (early-stage) faults in transformer insulation and windings, which generate dissolved gases in oil.

Unlike conventional rule-based methods (Duval Triangle, Rogers Ratio, Key Gas), which are rigid and often ambiguous, this project uses a data-driven ML approach to improve diagnostic accuracy and reliability.

To ensure realistic and controlled behavior, the dataset is physics-informed and generated using MATLAB simulations.

---

## Key Features

- Physics-based dataset generation using MATLAB
- Multi-class transformer fault classification (7 classes)
- Comparison of multiple machine learning models:
  - Random Forest
  - Decision Tree
  - XGBoost
- Feature engineering and ablation (raw vs ratio features)
- Benchmarking against traditional DGA techniques:
  - IEC Ratio Method
  - Duval Triangle
- Model explainability using SHAP
- Modular and scalable pipeline design
- API-ready backend for integration with UI or monitoring systems

---

## Fault Classes

- PD – Partial Discharge  
- D1 – Low Energy Discharge (Spark)  
- D2 – High Energy Discharge (Arc)  
- T1 – Low Temperature Thermal Fault (<300°C)  
- T2 – Medium Temperature Thermal Fault (300–700°C)  
- T3 – High Temperature Thermal Fault (>700°C)  
- Normal – Healthy transformer condition  

---

## Dataset

- Size: ~10,000 samples  
- Source: MATLAB-based physics simulation  
- Features:
  - H2, CH4, C2H6, C2H4, C2H2, CO, CO2
- Additional engineered ratios (used for comparison)

---

## Results Summary

| Metric | Result |
|------|--------|
| ML Accuracy | ~97% |
| Best Model | Random Forest |
| Weak Area | T1 vs Normal |
| Raw vs Ratio Features | Raw performs better |
| IEC Ratio Accuracy | ~29.6% |
| Duval Triangle Accuracy | ~35.4% |

---

## Installation & Setup

### 1. Clone the repository

git clone https://github.com/your-username/transformer-fault-ml.git
cd transformer-fault-ml

### 2. Create virtual environment

python -m venv venv

Activate:

Windows:
.\venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 3. Install dependencies

pip install -r requirements.txt

---

## Running the Project

python main.py

---

## License

Academic and research purposes only.
