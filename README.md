# Liver Stiffness SVM

Reproducible baseline machine-learning code for evaluating a linear support-vector machine on tabular liver-stiffness features.

The original repository was a small experiment prototype. This version keeps the same core idea—standardization, stratified cross-validation, and a linear SVM—while making it runnable from the command line and reusable with a real CSV dataset.

> **Important:** This is research software and a teaching baseline. It is not a clinical decision tool, has not been clinically validated, and must not be used to make patient-care decisions.

## Quick start

Create an environment and install the small dependency set:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the included deterministic demo:

```bash
python Liver_stiffness_SVM.py
```

The command prints a JSON report containing accuracy, ROC AUC, the confusion matrix, and dataset dimensions. Save the same report to disk with:

```bash
python Liver_stiffness_SVM.py --output reports/demo_metrics.json
```

## Using a CSV file

Pass a CSV containing numeric feature columns and one binary target column:

```text
age,platelet_count,stiffness_measurement,label
12,210,8.4,0
16,145,14.2,1
...
```

Run the experiment by naming the target column:

```bash
python Liver_stiffness_SVM.py \
  --data data/liver_features.csv \
  --target label \
  --folds 5 \
  --output reports/liver_svm_metrics.json
```

The loader expects finite numeric features and exactly two target classes. Missing values and categorical variables should be cleaned or encoded before running the baseline.

## Method

Each cross-validation training split fits its own `StandardScaler`, followed by a linear `SVC`. Out-of-fold predictions are then used to calculate the final metrics. Keeping scaling inside the scikit-learn pipeline prevents information from the test folds leaking into training.

The default classifier uses balanced class weights, which is a reasonable starting point when clinically relevant cohorts are not perfectly balanced. Hyperparameter tuning, calibration, external validation, subgroup analysis, and uncertainty estimates are intentionally left for a future study rather than hidden behind this baseline.

## Project layout

```text
.
├── Liver_stiffness_SVM.py       # CLI and reusable experiment functions
├── tests/
│   └── test_liver_stiffness_svm.py
├── requirements.txt
└── README.md
```

## Tests

After installing the dependencies, run:

```bash
python -m unittest discover -s tests -v
```

## Reproducibility

The demo dataset, fold shuffling, and SVM probability model use seed `42` by default. Change it with `--seed` when exploring sensitivity to random splits.

## License

This repository is provided for research and educational use. Add the license required by your lab or institution before distributing it externally.
