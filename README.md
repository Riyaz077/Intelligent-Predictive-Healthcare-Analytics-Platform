# Health-Intelligence

## AI-Powered Healthcare Intelligence and Disease Prediction

An educational, end-to-end healthcare analytics prototype for exploratory analysis, multi-disease classification, patient-level model predictions, model evaluation, and an interactive Streamlit dashboard.

> **Important:** This is a research/learning prototype, not a medical device or diagnostic service. It must not be used to make clinical decisions or replace a qualified healthcare professional. Predictions are dataset-dependent and require external clinical validation before any real-world use.

## Supported prediction tasks
1. Heart disease — UCI Heart Disease dataset (CSV supplied by the user).
2. Diabetes — Pima Indians Diabetes dataset (CSV supplied by the user).
3. Breast cancer — Wisconsin Diagnostic Breast Cancer dataset (CSV supplied by the user).

Each task is trained independently because the datasets have different features and target definitions.

## Features
- CSV upload and dataset inspection
- Missing-value handling, categorical encoding and scaling
- Logistic Regression, Random Forest, SVM, Gradient Boosting and XGBoost (optional)
- Stratified train/test split and cross-validation
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix and ROC curve
- Permutation feature importance (model-agnostic)
- Streamlit dashboard for training, evaluation and single-record prediction
- Unit tests for core preprocessing and evaluation utilities

## Repository layout
```text
.
├── dashboard/app.py
├── data/raw/                 # Place downloaded CSV files here (not committed)
├── docs/                     # Architecture, methodology and limitations
├── notebooks/                # Notebook workflow guides
├── reports/figures/          # Exported charts/screenshots
├── src/                      # Reusable Python modules
├── tests/                    # Unit tests
├── models/                   # Trained models (not committed by default)
├── requirements.txt
└── config.py
```

## Dataset setup
Download the datasets from their original sources and place CSV files in `data/raw/`:
- Heart disease: https://archive.ics.uci.edu/dataset/45/heart+disease
- Diabetes: https://archive.ics.uci.edu/dataset/34/diabetes
- Breast cancer: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

For the UCI heart dataset, prepare a CSV with one row per patient and a target column. The common Cleveland processed file uses `num` as the target; this project treats values greater than 0 as positive. For the Pima dataset, the target is usually `Outcome`. For Wisconsin Diagnostic Breast Cancer, use the `diagnosis` column (M/B) or a numeric target. Check each dataset's license and terms before redistribution. Do not commit patient-identifiable or sensitive data.

## Installation
Python 3.10–3.12 is recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run dashboard/app.py
```
Upload a CSV, choose the target column and positive class (if applicable), train models, inspect metrics, and optionally enter feature values for a demonstration prediction.

## Run tests
```bash
pytest -q
```

## Train from the command line
```bash
python -m src.train --csv data/raw/diabetes.csv --target Outcome --positive-class 1 --output models/diabetes.joblib
```

## Results
Do not publish example or invented scores as measured results. Add your own evaluation results after running the pipeline. Record dataset version, target definition, split strategy, random seed and metrics for reproducibility.

## Suggested screenshots
Save genuine screenshots of:
- Dashboard overview
- Dataset preview and target distribution
- Model comparison table
- Confusion matrix and ROC curve
- Feature importance
- Example prediction (clearly labelled as a demonstration)

## Responsible use and limitations
- Model outputs are not diagnoses or treatment recommendations.
- Dataset labels may be noisy, outdated, imbalanced or unrepresentative.
- Performance on a held-out test set does not establish clinical utility.
- Validate calibration, subgroup performance, fairness, data leakage and external generalization before any real-world research deployment.
- Do not enter identifiable patient data into this demo.

## Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai

