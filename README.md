# Fraud Detection Using Machine Learning

A capstone project that builds and evaluates machine learning models to detect fraudulent financial transactions, using the IEEE-CIS Fraud Detection dataset. The final delivered model is a LightGBM gradient-boosted classifier, selected for its strong discriminative performance and native support for explainability via feature importance and SHAP.

## Problem Statement

Traditional rule-based fraud detection systems used in the financial sector struggle to keep pace with evolving fraud patterns and often lack the transparency required for regulatory and compliance review. This project investigates whether a machine learning approach — specifically gradient-boosted trees — can improve on this by learning from historical transaction data while remaining interpretable enough for use by fraud analysts and compliance teams.

## Objectives

- Build a data pipeline covering exploratory data analysis, feature engineering, and model training on transaction data.
- Train and compare multiple candidate models (Logistic Regression, Random Forest, LightGBM, CatBoost) under a consistent, realistic evaluation setup.
- Select and tune a final model on the basis of AUC-ROC and AUC-PR, targeting a precision of at least 0.85 and a recall of at least 0.80 on a held-out validation set.
- Provide model interpretability through feature importance rankings and SHAP-based explanations.

## Dataset

- **Source:** [IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection) (Kaggle), originally released by Vesta Corporation and the IEEE Computational Intelligence Society.
- **Working sample:** 50,000 transactions sampled from the full training partition, for computational feasibility during iterative development.
- **Class balance:** ~3.59% fraud rate overall.
- **Split strategy:** chronological 80/20 split by `TransactionDT` (40,000 training / 10,000 validation), rather than a random split, to mirror real-world deployment where a model is trained on past data and evaluated on future transactions.

## Methodology

1. **Exploratory Data Analysis** — class imbalance, missingness audit, `TransactionAmt` distribution, and time structure.
2. **Data Preparation / Feature Engineering:**
   - Dropped 208 columns exceeding 75% missingness (mostly anonymised Vesta `V` columns, `D` timedelta columns, and `id_` identity columns).
   - Engineered temporal features (`DT_day`, `DT_hour`, `DT_dow`), transaction amount transforms (`TransactionAmt_log`, `TransactionAmt_decimal`), frequency encodings (`card1_freq`, `card2_freq`, `addr1_freq`, `P_emaildomain_freq`), a synthetic per-user identifier (`uid`) with per-user aggregates (`uid_amt_mean`, `uid_amt_std`, `uid_txn_count`, `amt_to_uid_mean_ratio`), and missingness indicator flags.
   - Final feature matrix: 241 features across 50,000 rows (40,000 train / 10,000 validation).
3. **Modelling** — four candidates trained and compared on the same validation split:
   - Logistic Regression (baseline)
   - Random Forest (`n_estimators=200`, `max_depth=12`, `min_samples_leaf=20`, `class_weight='balanced'`)
   - LightGBM (final model)
   - CatBoost
4. **Model Selection** — LightGBM selected for achieving the highest AUC-ROC and AUC-PR of the four candidates, while training substantially faster than CatBoost.
5. **Threshold Tuning** — F1-optimal decision threshold selected on the validation set.
6. **Explainability** — LightGBM gain-based feature importance and SHAP value analysis.

## Final Model

| | |
|---|---|
| **Algorithm** | LightGBM (`gbdt`, binary objective) |
| **Key hyperparameters** | `num_leaves=63`, `learning_rate=0.02`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=1`, `scale_pos_weight≈26.7` (computed) |
| **Training** | Early stopping on validation AUC (100 rounds patience), capped at 2,000 rounds; best iteration at round 463 |
| **Reproducibility** | `seed=42` |

## Results

Evaluated on a 10,000-transaction, time-based validation set (9,653 legitimate / 347 fraudulent):

| Metric | Result |
|---|---|
| AUC-ROC | 0.8870 |
| AUC-PR | 0.4855 |
| Precision (at F1-optimal threshold, 0.6237) | 0.64 |
| Recall (at F1-optimal threshold) | 0.41 |

LightGBM outperformed the Logistic Regression baseline and the other candidates on both AUC metrics. The model fell short of the project's original 0.85 precision / 0.80 recall targets at a single fixed threshold, indicating it is currently better suited to triaging transactions for human review than fully automated decisioning. See the project documentation for a full discussion of this gap and its likely causes (class imbalance, concept drift, and the absence of a formal hyperparameter search).

**Top contributing features:** Vesta's engineered counting features (`C1`, `C13`, `C14`, `C2`) dominate feature importance, alongside several features engineered in this project — `DT_day`, `uid_amt_mean`, `card1_freq`, and `P_emaildomain_freq` — confirming the engineered features carry independent predictive signal.

## Repository Structure

```
.
├── capStone.ipynb            # Initial pipeline: EDA, logistic regression, random forest baselines
├── Implementation.ipynb      # Full pipeline: EDA -> feature engineering -> model comparison -> 
│                              # threshold tuning -> feature importance -> SHAP explainability
├── training_dataset_sampled.csv   # 50,000-row sampled subset of the IEEE-CIS dataset
├── docs/
│   └── Final_Documentation.docx   # Full project write-up (introduction, literature review,
│                                   # methodology, results, recommendations)
└── README.md
```

> Adjust the structure above to match your actual repository layout before publishing.

## Getting Started

### Prerequisites

- Python 3.13
- `scikit-learn`, `lightgbm`, `catboost`, `pandas`, `numpy`, `shap`, `matplotlib`

### Installation

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
```

### Usage

Open `Implementation.ipynb` in Jupyter and run all cells to reproduce the full pipeline, from raw data through to the final model and SHAP explanations.

```bash
jupyter notebook Implementation.ipynb
```

## Limitations

- Trained on a static, anonymised public dataset rather than live transaction data.
- No formal hyperparameter search was conducted; the reported results likely understate the model's full potential.
- Several raw features (the Vesta `V` columns) are semantically opaque, limiting how far feature engineering can go in extracting additional signal.
- Results reflect fraud patterns present in the dataset's collection period and may not generalise to current fraud behaviour (concept drift).

## Recommendations

- Adopt LightGBM as the starting candidate for further development.
- Prioritise identity-resolution and per-user behavioural features in future iterations.
- Treat the current recall figure as a baseline to improve on, via hyperparameter search and threshold recalibration.
- Deploy as a triage/ranking tool supporting human reviewers rather than a fully automated decision system.

## Author

Khwezi Mabija — Data Science Capstone Project
