"""
=============================================================
  Customer Churn Prediction - Complete Analysis Pipeline
=============================================================
Dataset : Telco Customer Churn (IBM Sample Dataset)
Author  : Data Science Project
Skills  : EDA · Feature Engineering · ML Classification · Evaluation
=============================================================
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, pickle, json, os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

warnings.filterwarnings('ignore')
os.makedirs('plots', exist_ok=True)
print("✅ Libraries loaded")

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(f"\n📂 Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ─────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

print("\n📊 Data Types:")
print(df.dtypes)

print("\n🔍 Missing Values:")
print(df.isnull().sum())

print("\n📈 Churn Distribution:")
print(df['Churn'].value_counts())
print(f"Churn Rate: {(df['Churn']=='Yes').mean():.2%}")

print("\n📉 Numerical Feature Stats:")
print(df.describe())

# ─────────────────────────────────────────────────────────────
# 3. DATA CLEANING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: DATA CLEANING")
print("="*60)

df_clean = df.copy()

# Fix TotalCharges – has spaces as strings for new customers
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
blanks = df_clean['TotalCharges'].isnull().sum()
print(f"  TotalCharges blank entries (new customers): {blanks}")
df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)

# Drop customerID (not a feature)
df_clean.drop('customerID', axis=1, inplace=True)

# Encode target
df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)

print(f"  Dataset after cleaning: {df_clean.shape}")
print(f"  Remaining nulls: {df_clean.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: FEATURE ENGINEERING")
print("="*60)

# Average spend per month (avoids total-charge bias for short-tenure customers)
df_clean['AvgMonthlySpend'] = df_clean['TotalCharges'] / (df_clean['tenure'] + 1)

# Count of value-add services (security, backup, tech support)
df_clean['HasMultipleServices'] = (
    (df_clean['OnlineSecurity'] == 'Yes').astype(int) +
    (df_clean['OnlineBackup']   == 'Yes').astype(int) +
    (df_clean['TechSupport']    == 'Yes').astype(int)
)

# Tenure bucketed into lifecycle stages
df_clean['TenureGroup'] = pd.cut(
    df_clean['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
)

print("  ✅ Added: AvgMonthlySpend, HasMultipleServices, TenureGroup")

# ─────────────────────────────────────────────────────────────
# 5. ENCODING
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: LABEL ENCODING")
print("="*60)

cat_cols = df_clean.select_dtypes(include='object').columns.tolist() + ['TenureGroup']
print(f"  Encoding {len(cat_cols)} categorical columns: {cat_cols}")

for col in cat_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

print("  ✅ Encoding complete")

# ─────────────────────────────────────────────────────────────
# 6. MODEL PREPARATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: TRAIN/TEST SPLIT")
print("="*60)

X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn']

# Impute any remaining NaN
imputer = SimpleImputer(strategy='median')
X_imp   = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imp, y, test_size=0.20, random_state=42, stratify=y
)

scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)

print(f"  Train set: {X_train.shape}  |  Test set: {X_test.shape}")
print(f"  Churn in train: {y_train.mean():.2%}  |  Churn in test: {y_test.mean():.2%}")

# ─────────────────────────────────────────────────────────────
# 7. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: MODEL TRAINING")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    uses_scale = name == 'Logistic Regression'
    Xtr = X_train_sc if uses_scale else X_train
    Xte = X_test_sc  if uses_scale else X_test

    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]

    results[name] = {
        'model'    : model,
        'preds'    : preds,
        'probs'    : probs,
        'acc'      : accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall'   : recall_score(y_test, preds),
        'f1'       : f1_score(y_test, preds),
        'auc'      : roc_auc_score(y_test, probs),
    }

    # 5-fold CV AUC
    cv_auc = cross_val_score(
        model.__class__(**model.get_params()),
        Xtr, y_train, cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='roc_auc', n_jobs=-1
    ).mean()

    print(f"\n  {name}")
    print(f"    Accuracy : {results[name]['acc']:.4f}")
    print(f"    AUC      : {results[name]['auc']:.4f}")
    print(f"    CV-AUC   : {cv_auc:.4f}")
    print(f"    Precision: {results[name]['precision']:.4f}  Recall: {results[name]['recall']:.4f}  F1: {results[name]['f1']:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name} (AUC={best['auc']:.4f})")
print("\n  Classification Report:")
print(classification_report(y_test, best['preds'], target_names=['Retained','Churned']))

# ─────────────────────────────────────────────────────────────
# 8. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
pickle.dump(best['model'],  open('model.pkl', 'wb'))
pickle.dump(scaler,         open('scaler.pkl', 'wb'))
pickle.dump(imputer,        open('imputer.pkl', 'wb'))
pickle.dump(list(X.columns),open('feature_cols.pkl', 'wb'))

summary = {
    'best_model': best_name,
    'best_acc'  : best['acc'],
    'best_auc'  : best['auc'],
    'metrics'   : {n: {'accuracy': r['acc'], 'auc': r['auc']} for n, r in results.items()}
}
json.dump(summary, open('model_metrics.json','w'), indent=2)
print("\n✅ Artifacts saved: model.pkl, scaler.pkl, imputer.pkl, feature_cols.pkl, model_metrics.json")

print("\n🎉 Pipeline complete! Run  streamlit run app.py  to launch the dashboard.")
