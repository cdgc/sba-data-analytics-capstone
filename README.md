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
| `processed_data/` | Two pipeline inputs (`analysis_sample.csv`, `model_data_with_split.csv`) and the consolidated `interim_report_evidence.xlsx` workbook. |
| `code/` | Three task-focused Python analysis notebooks, stored directly in this folder. |
| `screenshots/` | Report figures, dataset preview, and project-structure evidence. |
| `ml_models/` | Serialized baseline and final models; local-only and excluded from version control. |
| `templates_checklist/` | Course report templates and checklists; retained locally. |

## Analysis workflow

| Notebook | Work performed | Report use |
|---|---|---|
| `code/01_data_preparation_and_eda.ipynb` | Resolves the project data folder, inventories source files, documents selected fields, draws a stratified analysis sample, estimates sample-size needs for each RQ, and saves the report evidence workbook and figures. | Source/data description, data dictionary, cleaning log, sample, sample-size plan, and EDA for the interim report. |
| `code/02_hypothesis_tests_and_baselines.ipynb` | Runs RQ1/RQ2 tests, fits the RQ3 logistic model with training-only imputation, creates the stratified split, and evaluates preliminary classification baselines on validation data. | Statistical analysis and preliminary modeling for the interim report. |
| `code/03_model_tuning_and_validation.ipynb` | Tunes Random Forest/XGBoost on training folds, selects a model using validation data, evaluates once on the held-out test set, and computes SHAP explanations. | Deferred until the final-report phase; not required for the interim report. |

The notebooks contain Python analysis code with short usage notes, not report-drafting sections. Write the interim report, final report, recommendations, and presentation in their required Word/PowerPoint templates.

The synopsis proposes a stratified sample of 50,000 loans, a stratified 70/15/15 train/validation/test split, and comparisons using logistic regression, Random Forest, and XGBoost. Treat these as planned choices and verify feasibility and assumptions before relying on them. Test data must not be used for feature selection, hyperparameter tuning, or threshold selection.

## Setup

Install the Python dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

For the interim report, run notebooks 01 and 02 in order. Notebook 01 creates `processed_data/analysis_sample.csv`, starts `processed_data/interim_report_evidence.xlsx`, and saves charts and report evidence in `screenshots/`. Notebook 02 creates `processed_data/model_data_with_split.csv`, adds hypothesis-test and baseline results to the same workbook, and stores serialized models in `ml_models/`. Notebook 03 is a later-phase workflow; do not run it until final model tuning and held-out testing are required. It will append final results to the evidence workbook and save its models and figures in the same separate folders.

The processed CSVs are pipeline inputs, the workbook is the consolidated report-table output, and model binaries are reproducibility artifacts rather than report deliverables. Source CSVs are processed in chunks. Keep source files unchanged in `raw_data/`; the original FOIA CSVs, course templates, and serialized models remain local. The interim report still needs to be written in the supplied template, including its literature review, APA citations, interpretations, and completed title-page details. Review outputs and assumptions before using any result in the report. Records are not deduplicated on `LocationID`, which the SBA data dictionary defines as a lender identifier, not a unique loan identifier.
