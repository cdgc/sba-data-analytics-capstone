# SBA Small Business Loan Default Analysis

This capstone studies factors associated with default in the U.S. Small Business Administration (SBA) 7(a) and 504 guaranteed loan programs, using public FOIA loan data.

## Research questions

1. Is business age associated with loan outcome?
2. Does loan term differ between paid-in-full and charged-off loans?
3. Which financial and operational characteristics are associated with default?
4. Do Random Forest or XGBoost improve classification over logistic regression, and which features are most informative?

The planned binary outcome is `CHGOFF` (default) versus `PIF` (paid in full). Confirm status mappings and exclusions against the source data dictionary before analysis.

## Project folders

| Folder | Purpose |
|---|---|
| `raw_data/` | Original SBA 7(a)/504 FOIA files and data dictionary. Treat these as read-only. |
| `processed_data/` | Cleaned/sampled data, analysis tables, plots, split data, and model artifacts. |
| `code/` | Three task-focused Python analysis notebooks, stored directly in this folder. |
| `screenshots/` | Dataset/tree screenshots and other report or presentation evidence. |

## Analysis workflow

| Notebook | Work performed | Report use |
|---|---|---|
| `code/01_data_preparation_and_eda.ipynb` | Inventories SBA files, draws a stratified sample in chunks, standardizes analysis fields, and saves data-quality and EDA outputs. | Dataset description, cleaning, sample, and EDA work for the interim report. |
| `code/02_hypothesis_tests_and_baselines.ipynb` | Runs RQ1/RQ2 tests, fits RQ3 logistic regression, creates the stratified split, and evaluates preliminary classification baselines on validation data. | Statistical analysis and preliminary modeling for the interim report. |
| `code/03_model_tuning_and_validation.ipynb` | Tunes Random Forest/XGBoost on training folds, selects a model using validation data, evaluates once on the held-out test set, and computes SHAP explanations. | Final report and presentation results. |

The notebooks contain Python analysis code with short usage notes, not report-drafting sections. Write the interim report, final report, recommendations, and presentation in their required Word/PowerPoint templates.

The synopsis proposes a stratified sample of 50,000 loans, a stratified 70/15/15 train/validation/test split, and comparisons using logistic regression, Random Forest, and XGBoost. Treat these as planned choices and verify feasibility and assumptions before relying on them. Test data must not be used for feature selection, hyperparameter tuning, or threshold selection.

## Setup

Install the Python dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Run notebooks in numeric order. Notebook 01 writes `analysis_sample.csv`; notebook 02 writes `model_data_with_split.csv`, baseline models, and test/validation tables; notebook 03 writes tuned models, final test metrics, figures, and SHAP feature importance. Keep source files unchanged in `raw_data/`; derived data and artifacts go to `processed_data/`, and requested evidence goes to `screenshots/`. Review outputs and assumptions before using any result in a report. Source CSVs are processed in chunks. Records are not deduplicated on `LocationID`, which the SBA data dictionary defines as a lender identifier, not a unique loan identifier.
