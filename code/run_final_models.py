import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300

# ---------------------------------------------------------
# Step 1: Load and Stratify Raw Data (N = 50,000)
# ---------------------------------------------------------
DATA_DIR = "../raw_data"
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

print("=" * 70)
print("1. INGESTING DATA AND BUILDING STRATIFIED SAMPLE (N = 50,000)")
print("=" * 70)

# Target columns for loan evaluation
cols_needed = [
    'LoanStatus', 'BusinessAge', 'TermInMonths', 'GrossApproval',
    'JobsSupported', 'CollateralInd', 'NaicsCode', 'FranchiseCode',
    'Program', 'ApprovalFY'
]

raw_chunks = []
for f in csv_files:
    # Read relevant columns in chunks
    df_temp = pd.read_csv(f, low_memory=False)
    # Filter matching columns
    matched_cols = [c for c in cols_needed if c in df_temp.columns]
    df_filtered = df_temp[matched_cols].dropna(subset=['LoanStatus']).copy()
    
    # Normalize LoanStatus
    df_filtered['LoanStatus'] = df_filtered['LoanStatus'].astype(str).str.strip().str.upper()
    # Filter to resolved loans only
    resolved = df_filtered[df_filtered['LoanStatus'].isin(['CHGOFF', 'PIF', 'P I F'])].copy()
    raw_chunks.append(resolved)

full_resolved_df = pd.concat(raw_chunks, ignore_index=True)
full_resolved_df['LoanStatus'] = full_resolved_df['LoanStatus'].replace({'P I F': 'PIF'})
print(f"Total resolved loans in database: {len(full_resolved_df):,}")

# Stratified random sample of 50,000 based on LoanStatus
sample_df = full_resolved_df.groupby('LoanStatus', group_keys=False).apply(
    lambda x: x.sample(n=int(50000 * len(x) / len(full_resolved_df)), random_state=42)
).reset_index(drop=True)

# Adjust to exactly 50,000 if rounding caused +/- 1 record
if len(sample_df) < 50000:
    extra = full_resolved_df.sample(50000 - len(sample_df), random_state=42)
    sample_df = pd.concat([sample_df, extra], ignore_index=True)
elif len(sample_df) > 50000:
    sample_df = sample_df.sample(50000, random_state=42).reset_index(drop=True)

print(f"Stratified Sample Size: {len(sample_df):,}")
print("Target Class Distribution:")
print(sample_df['LoanStatus'].value_counts(normalize=True) * 100)

# ---------------------------------------------------------
# Step 2: Data Cleaning Log Diagnostics
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("2. DATA CLEANING AUDIT & PREPROCESSING")
print("=" * 70)

# A. Duplicate audit
dup_count = sample_df.duplicated().sum()
sample_df = sample_df.drop_duplicates().reset_index(drop=True)
print(f"Duplicate rows removed: {dup_count} ({(dup_count/50000)*100:.2f}%)")

# B. Missingness detection
print("\nMissing values before imputation:")
missing_series = sample_df.isnull().sum()
missing_pct = (sample_df.isnull().sum() / len(sample_df)) * 100
missing_report = pd.DataFrame({'Missing Count': missing_series, 'Percent (%)': missing_pct})
print(missing_report[missing_report['Missing Count'] > 0])

# C. Handling Missing Values and Data Types
# Clean GrossApproval to float
if sample_df['GrossApproval'].dtype == object:
    sample_df['GrossApproval'] = sample_df['GrossApproval'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
sample_df['GrossApproval'] = pd.to_numeric(sample_df['GrossApproval'], errors='coerce')
sample_df['GrossApproval'] = sample_df['GrossApproval'].fillna(sample_df['GrossApproval'].median())

# Clean TermInMonths
sample_df['TermInMonths'] = pd.to_numeric(sample_df['TermInMonths'], errors='coerce')
sample_df['TermInMonths'] = sample_df['TermInMonths'].fillna(sample_df['TermInMonths'].median())
# Filter invalid 0 term records
sample_df = sample_df[sample_df['TermInMonths'] > 0].copy()

# Clean JobsSupported
sample_df['JobsSupported'] = pd.to_numeric(sample_df['JobsSupported'], errors='coerce').fillna(1)

# Clean Categoricals
sample_df['BusinessAge'] = sample_df['BusinessAge'].fillna('Existing Business').astype(str)
sample_df['CollateralInd'] = sample_df['CollateralInd'].fillna('N').astype(str).str.upper()

# D. Feature Engineering
sample_df['NaicsSector'] = sample_df['NaicsCode'].astype(str).str[:2]
sample_df['FranchiseInd'] = sample_df['FranchiseCode'].astype(str).apply(
    lambda x: 0 if x in ['0', '99999', 'nan', ''] else 1
)
sample_df['Target_Default'] = (sample_df['LoanStatus'] == 'CHGOFF').astype(int)

# ---------------------------------------------------------
# Step 3: Statistical Hypothesis Testing (RQ1 & RQ2)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("3. HYPOTHESIS TESTING RESULTS (RQ1 & RQ2)")
print("=" * 70)

# RQ1: Chi-Square Test (BusinessAge vs LoanStatus)
print("--- Research Question 1: Chi-Square Test of Independence ---")
contingency_table = pd.crosstab(sample_df['BusinessAge'], sample_df['LoanStatus'])
print("Contingency Table:")
print(contingency_table)

chi2, p_val_rq1, dof, expected = stats.chi2_contingency(contingency_table)
n_total = contingency_table.sum().sum()
min_dim = min(contingency_table.shape) - 1
cramers_v = np.sqrt(chi2 / (n_total * min_dim))

print(f"Chi-Square Statistic: {chi2:.4f}")
print(f"Degrees of Freedom: {dof}")
print(f"p-value: {p_val_rq1:.4e}")
print(f"Cramer's V Effect Size: {cramers_v:.4f}")

# RQ2: Welch's t-Test (TermInMonths by LoanStatus)
print("\n--- Research Question 2: Independent Welch's t-Test ---")
term_chgoff = sample_df[sample_df['LoanStatus'] == 'CHGOFF']['TermInMonths'].dropna()
term_pif = sample_df[sample_df['LoanStatus'] == 'PIF']['TermInMonths'].dropna()

mean_chgoff, std_chgoff = term_chgoff.mean(), term_chgoff.std()
mean_pif, std_pif = term_pif.mean(), term_pif.std()

t_stat, p_val_rq2 = stats.ttest_ind(term_chgoff, term_pif, equal_var=False)

# Cohen's d calculation
pooled_std = np.sqrt(((len(term_chgoff) - 1) * std_chgoff**2 + (len(term_pif) - 1) * std_pif**2) / (len(term_chgoff) + len(term_pif) - 2))
cohens_d = (mean_chgoff - mean_pif) / pooled_std

print(f"CHGOFF (Default) Mean Term: {mean_chgoff:.2f} months (SD = {std_chgoff:.2f})")
print(f"PIF (Paid) Mean Term: {mean_pif:.2f} months (SD = {std_pif:.2f})")
print(f"Welch's t-Statistic: {t_stat:.4f}")
print(f"p-value: {p_val_rq2:.4e}")
print(f"Cohen's d Effect Size: {cohens_d:.4f}")

# ---------------------------------------------------------
# Step 4: Generate EDA Visualizations for Report
# ---------------------------------------------------------
os.makedirs("../screenshots", exist_ok=True)

# Figure 1: BusinessAge vs Default Rate
plt.figure(figsize=(10, 6))  
age_default_rate = sample_df.groupby('BusinessAge')['Target_Default'].mean().reset_index()

age_default_rate = age_default_rate.sort_values('Target_Default', ascending=False)

sns.barplot(data=age_default_rate, x='BusinessAge', y='Target_Default', palette='Blues_r')
plt.title("Figure 1: Default Rate by Business Lifecycle Stage", fontsize=11, fontweight='bold')
plt.ylabel("Default Rate (Proportion)")
plt.xlabel("Business Age Category")

plt.xticks(rotation=45, ha='right')

plt.tight_layout()  # Adjusts margins so the rotated labels are not cut off
plt.savefig("../screenshots/figure1_business_age_default.png")
plt.close()

# Figure 2: Boxplot of TermInMonths by LoanStatus
plt.figure(figsize=(8, 5))
sns.boxplot(data=sample_df, x='LoanStatus', y='TermInMonths', palette='Set2')
plt.title("Figure 2: Distribution of Loan Term (Months) by Loan Disposition", fontsize=11, fontweight='bold')
plt.xlabel("Loan Status")
plt.ylabel("Term (Months)")

plt.xticks(rotation=0)  

plt.tight_layout()
plt.savefig("../screenshots/figure2_loan_term_distribution.png")
plt.close()

# Figure 3: Distribution of Log(GrossApproval)
plt.figure(figsize=(8, 5))
sample_df['Log_GrossApproval'] = np.log(sample_df['GrossApproval'])
sns.histplot(data=sample_df, x='Log_GrossApproval', hue='LoanStatus', kde=True, bins=40, palette='tab10')
plt.title("Figure 3: Distribution of Log Loan Principal by Loan Disposition", fontsize=12, fontweight='bold')
plt.xlabel("Log(Gross Loan Approval Amount)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("../screenshots/figure3_log_gross_approval.png")
plt.close()

print("\nAll figures exported successfully to '../screenshots/' folder.")

#########Model Training 
# ---------------------------------------------------------
# Step 5: RQ3 - Multivariate Binary Logistic Regression & VIF
# ---------------------------------------------------------
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
import xgboost as xgb
import shap

print("\n" + "=" * 70)
print("4. RQ3: MULTIVARIATE BINARY LOGISTIC REGRESSION & VIF DIAGNOSTICS")
print("=" * 70)

# Save cleaned analytical sample to ../processed_data/
os.makedirs("../processed_data", exist_ok=True)
sample_df.to_csv("../processed_data/analytical_sample.csv", index=False)

# Prepare model-ready engineered features matching Steps 1-4
sample_df['Log_GrossApproval'] = np.log1p(sample_df['GrossApproval'].clip(lower=1))
sample_df['Log_JobsSupported'] = np.log1p(sample_df['JobsSupported'].clip(lower=0))
sample_df['Collateral_Bin'] = (sample_df['CollateralInd'] == 'Y').astype(int)
sample_df['Program_7a'] = sample_df['Program'].astype(str).str.contains('7').astype(int)
sample_df['Is_New_Business'] = sample_df['BusinessAge'].astype(str).str.contains(
    'New|Startup|Less than 1|at least 2', case=False, regex=True
).astype(int)

# Stratified 70% Train / 15% Validation / 15% Holdout Test Split
train_df, temp_df = train_test_split(
    sample_df, test_size=0.30, stratify=sample_df['Target_Default'], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['Target_Default'], random_state=42
)
print(f"Partition Sizes -> Train (70%): {len(train_df):,} | Val (15%): {len(val_df):,} | Test (15%): {len(test_df):,}")

rq3_cols = [
    'Log_GrossApproval', 'TermInMonths', 'Log_JobsSupported',
    'Collateral_Bin', 'Is_New_Business', 'FranchiseInd', 'Program_7a'
]

X_train_rq3 = sm.add_constant(train_df[rq3_cols].astype(float))
y_train_rq3 = train_df['Target_Default'].astype(int)

vif_df = pd.DataFrame({
    "Feature": rq3_cols,
    "VIF": [variance_inflation_factor(X_train_rq3.values, i + 1) for i in range(len(rq3_cols))]
}).set_index("Feature")

logit_sm = sm.Logit(y_train_rq3, X_train_rq3).fit(disp=False)

rq3_table = pd.DataFrame({
    'Coef (Beta)': logit_sm.params,
    'Std.Err': logit_sm.bse,
    'Wald z': logit_sm.tvalues,
    'p-value': logit_sm.pvalues,
    'Odds Ratio (OR)': np.exp(logit_sm.params),
    '95% CI Lower': np.exp(logit_sm.conf_int()[0]),
    '95% CI Upper': np.exp(logit_sm.conf_int()[1])
}).join(vif_df)

print("\n--- RQ3 Multivariate Logistic Regression Table ---")
print(rq3_table.round(4).to_string())
print(f"\nLog-Likelihood: {logit_sm.llf:.2f} | LLR p-value: {logit_sm.llr_pvalue:.4e} | Pseudo R2: {logit_sm.prsquared:.4f}")

# ---------------------------------------------------------
# Step 6: RQ4 - Machine Learning Benchmarking (Logit vs RF vs XGBoost)
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("5. RQ4: MODEL PERFORMANCE COMPARISON (TRAIN, VALIDATION & TEST)")
print("=" * 70)

num_features = ['Log_GrossApproval', 'TermInMonths', 'Log_JobsSupported',
                'Collateral_Bin', 'FranchiseInd', 'Program_7a']
cat_features = ['BusinessAge', 'NaicsSector']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

X_train_proc = preprocessor.fit_transform(train_df[num_features + cat_features])
y_train = train_df['Target_Default'].values

X_val_proc = preprocessor.transform(val_df[num_features + cat_features])
y_val = val_df['Target_Default'].values

X_test_proc = preprocessor.transform(test_df[num_features + cat_features])
y_test = test_df['Target_Default'].values

feature_names = (
    num_features +
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features))
)

neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=20,
        class_weight='balanced', n_jobs=-1, random_state=42
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=neg_pos_ratio, eval_metric='logloss',
        random_state=42, n_jobs=-1
    )
}

perf_rows = []
test_preds = {}
test_probs = {}

for name, clf in models.items():
    clf.fit(X_train_proc, y_train)
    for part_name, X_p, y_p in [("Train (70%)", X_train_proc, y_train),
                                ("Validation (15%)", X_val_proc, y_val),
                                ("Test (15%)", X_test_proc, y_test)]:
        preds = clf.predict(X_p)
        probs = clf.predict_proba(X_p)[:, 1]
        perf_rows.append({
            "Model": name,
            "Partition": part_name,
            "Accuracy": accuracy_score(y_p, preds),
            "Precision": precision_score(y_p, preds, zero_division=0),
            "Recall": recall_score(y_p, preds, zero_division=0),
            "F1-Score": f1_score(y_p, preds, zero_division=0),
            "ROC-AUC": roc_auc_score(y_p, probs),
            "PR-AUC": average_precision_score(y_p, probs)
        })
        if part_name == "Test (15%)":
            test_preds[name] = preds
            test_probs[name] = probs

perf_df = pd.DataFrame(perf_rows)
print("\n--- Model Performance Summary Table ---")
print(perf_df.round(4).to_string(index=False))

print("\n--- Holdout Test Set Confusion Matrices (TN, FP, FN, TP) ---")
for name in models.keys():
    cm = confusion_matrix(y_test, test_preds[name])
    tn, fp, fn, tp = cm.ravel()
    print(f"{name:22s} -> TN: {tn:5d} | FP: {fp:5d} | FN: {fn:5d} | TP: {tp:5d}")

# ---------------------------------------------------------
# Step 7: Export Model Evaluation & SHAP Visualizations to ../screenshots/
# ---------------------------------------------------------
# Figure 5: ROC and Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "XGBoost": "#d62728"}

for name in models.keys():
    fpr, tpr, _ = roc_curve(y_test, test_probs[name])
    auc_val = roc_auc_score(y_test, test_probs[name])
    axes[0].plot(fpr, tpr, lw=2, color=colors[name], label=f"{name} (AUC = {auc_val:.3f})")

    prec, rec, _ = precision_recall_curve(y_test, test_probs[name])
    pr_val = average_precision_score(y_test, test_probs[name])
    axes[1].plot(rec, prec, lw=2, color=colors[name], label=f"{name} (PR-AUC = {pr_val:.3f})")

axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label="Random Chance (0.500)")
axes[0].set_title("A. Receiver Operating Characteristic (ROC) Curve", fontsize=11, fontweight='bold')
axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)")
axes[0].legend(loc="lower right", fontsize=9)

axes[1].set_title("B. Precision-Recall (PR) Curve", fontsize=11, fontweight='bold')
axes[1].set_xlabel("Recall (Sensitivity)")
axes[1].set_ylabel("Precision (Positive Predictive Value)")
axes[1].legend(loc="upper right", fontsize=9)

plt.suptitle("Figure 5: Out-of-Sample ROC and Precision-Recall Curves (Holdout Test Set)", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("../screenshots/figure5_roc_pr_curves.png")
plt.close()

# Figure 6: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for idx, name in enumerate(models.keys()):
    cm = confusion_matrix(y_test, test_preds[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[idx],
                xticklabels=['Pred PIF (0)', 'Pred CHGOFF (1)'],
                yticklabels=['Actual PIF (0)', 'Actual CHGOFF (1)'])
    axes[idx].set_title(f"{name}", fontsize=11, fontweight='bold')
    axes[idx].set_ylabel("Actual Loan Status" if idx == 0 else "")
    axes[idx].set_xlabel("Predicted Loan Status")

plt.suptitle("Figure 6: Holdout Test Set Confusion Matrices Across Models", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("../screenshots/figure6_confusion_matrices.png")
plt.close()

# Figure 7: SHAP Feature Importance
xgb_clf = models["XGBoost"]
try:
    explainer = shap.TreeExplainer(xgb_clf)
    shap_vals = explainer.shap_values(X_test_proc[:1000])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    mean_abs_imp = np.abs(shap_vals).mean(axis=0)
    imp_label = "Mean |SHAP Value| (Impact on Default Log-Odds)"
except Exception:
    mean_abs_imp = xgb_clf.feature_importances_
    imp_label = "Relative Feature Importance Score"

imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': mean_abs_imp
}).sort_values('Importance', ascending=False).head(10)

print("\n--- Top 10 Predictive Features Driving Loan Default (RQ4) ---")
print(imp_df.round(4).to_string(index=False))

plt.figure(figsize=(9, 5))
sns.barplot(data=imp_df, x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title("Figure 7: Top 10 Predictive Features Driving Loan Default (SHAP / Tree Importance)", fontsize=11, fontweight='bold')
plt.xlabel(imp_label)
plt.ylabel("Predictor Feature")
plt.tight_layout()
plt.savefig("../screenshots/figure7_feature_importance_shap.png")
plt.close()

print("\nAll final model figures (Figures 5, 6, and 7) exported to '../screenshots/' folder.")
