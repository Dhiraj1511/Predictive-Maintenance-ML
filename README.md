# Intelligent Predictive Maintenance and Machine Health Monitoring System using Machine Learning

## Project overview
This project uses machine sensor readings to estimate whether a machine failure may occur. It also explores sensor patterns, compares machine-learning models, highlights influential features, detects unusual sensor combinations, and provides a Streamlit dashboard for interactive predictions.

**Dataset:** UCI AI4I 2020 Predictive Maintenance Dataset  
Official page: https://archive.ics.uci.edu/dataset/601/ai4i  
DOI: https://doi.org/10.24432/C5HS5C

The AI4I dataset is synthetic and contains 10,000 examples. It is designed to reflect predictive-maintenance data encountered in industry. Be transparent about this in the report: it is a useful benchmark, but it is not real production-machine history.

## Main features
1. Load the official dataset automatically.
2. Inspect missing values, duplicate rows, target distribution, and summary statistics.
3. Create EDA plots and a numeric correlation heatmap.
4. Train Logistic Regression and Random Forest models.
5. Compare accuracy, precision, recall, F1, ROC-AUC, and average precision.
6. Prevent target leakage by excluding UDI/Product ID and failure-mode labels (TWF, HDF, PWF, OSF, RNF) from the main classifier.
7. Save a model chosen by failure recall, with F1 as a tie-breaker.
8. Compute permutation feature importance.
9. Fit an Isolation Forest for unsupervised anomaly detection.
10. Run an interactive Streamlit dashboard for single-record and batch predictions.

## Requirements
- Python 3.10 recommended
- Internet access for the first dataset download
- VS Code is recommended but not required

## Setup in VS Code (Windows)
Open this folder in VS Code. In the terminal run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

## Train the project
From the project root folder:

```powershell
python src/train_model.py
```

The first run downloads the UCI dataset into `data/raw/`. It creates:
- `data/processed/ai4i_clean.csv`
- `outputs/data_quality.json`
- `outputs/model_comparison.csv`
- `outputs/classification_reports.json`
- `outputs/test_predictions.csv`
- `outputs/anomaly_results.csv`
- `outputs/feature_importance.csv`
- charts under `outputs/figures/`
- trained models under `models/`

If the download fails, download the CSV from the official UCI page and save it as `data/raw/ai4i2020.csv`.

## Launch the dashboard
After training finishes:

```powershell
python -m streamlit run app.py
```

Your browser will open the local dashboard. The dashboard supports one-machine predictions and batch CSV predictions.

## How the ML workflow works
1. **Target:** `Machine failure`, where 0 means no failure and 1 means failure.
2. **Input features:** product type and sensor readings.
3. **Split:** 80% training and 20% test, stratified so the class proportions are similar.
4. **Preprocessing:** median imputation and standard scaling for numeric columns; most-frequent imputation and one-hot encoding for categorical columns.
5. **Models:** Logistic Regression and Random Forest.
6. **Evaluation:** precision, recall, F1, ROC-AUC, and average precision, in addition to accuracy.
7. **Model selection:** prioritize recall for the failure class, then F1. This is sensible for a maintenance screening tool because missing a failure may be costly; nevertheless, precision and false alarms must also be considered.
8. **Explainability:** permutation importance estimates how much the evaluation score changes when a feature is shuffled. It is a global model explanation, not proof of causation.
9. **Anomaly detection:** Isolation Forest flags unusual sensor combinations. An anomaly is not automatically a machine failure.
10. **Dashboard:** shows estimated probability and a user-selectable alert threshold.

## Important limitations
- AI4I is synthetic, not a live factory dataset.
- A model prediction is a probability estimate, not a guarantee.
- The dataset does not provide reliable sequential operating history for each machine, so this project does **not** claim a validated Remaining Useful Life (RUL) prediction.
- Anomaly detection and failure classification answer different questions.
- Thresholds should be selected using validation data and the cost of missed failures versus false alarms.
- This is an academic demonstration, not a certified industrial safety system.

## Suggested report structure
1. Abstract
2. Introduction and problem statement
3. Dataset description
4. Data preprocessing and exploratory analysis
5. Methodology and model design
6. Evaluation metrics and results
7. Explainability and anomaly detection
8. Dashboard and use case
9. Limitations
10. Conclusion and future work

## Source
UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset: https://archive.ics.uci.edu/dataset/601/ai4i
