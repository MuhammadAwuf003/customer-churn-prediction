# 📊 Customer Churn Prediction

> Predict whether a Telco customer is likely to churn using machine learning.

---

## 🗂️ Project Structure

```
churn_project/
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 rows × 21 cols)
├── train.py                                 # Full ML pipeline script
├── app.py                                   # Streamlit dashboard
├── requirements.txt                         # Python dependencies
├── model.pkl                                # Trained model (Logistic Regression)
├── scaler.pkl                               # StandardScaler
├── imputer.pkl                              # SimpleImputer
├── feature_cols.pkl                         # Feature column order
├── model_metrics.json                       # Accuracy & AUC scores
└── plots/                                   # EDA & evaluation charts
    ├── churn_distribution.png
    ├── tenure_churn.png
    ├── monthly_charges_churn.png
    ├── contract_churn.png
    ├── correlation_heatmap.png
    ├── roc_curves.png
    └── confusion_matrix.png
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📌 Dataset

**IBM Telco Customer Churn** — 7,043 customers, 21 features including:

| Category       | Features |
|----------------|----------|
| Demographics   | gender, SeniorCitizen, Partner, Dependents |
| Services       | PhoneService, InternetService, OnlineSecurity, StreamingTV, … |
| Billing        | Contract, PaymentMethod, MonthlyCharges, TotalCharges |
| **Target**     | **Churn** (Yes / No) |

Churn rate: **26.5%** (class imbalance handled via stratified split)

---

## 🔬 Methodology

### Data Cleaning
- Converted `TotalCharges` from string → numeric (11 blank entries for new customers imputed with median)
- Dropped `customerID` (non-predictive identifier)
- Binary-encoded `Churn` target (Yes → 1, No → 0)

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `AvgMonthlySpend` | `TotalCharges / (tenure + 1)` — normalizes spend across tenure |
| `HasMultipleServices` | Count of OnlineSecurity + OnlineBackup + TechSupport |
| `TenureGroup` | Bucketed tenure: 0-1yr, 1-2yr, 2-4yr, 4-6yr |

### Models Compared
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 0.8034 | **0.8457** ✅ |
| Gradient Boosting | 0.8034 | 0.8432 |
| Random Forest | 0.7842 | 0.8206 |

**Best Model: Logistic Regression** (AUC = 0.846)

---

## 🎯 Key Findings

1. **Contract type** is the strongest churn predictor — month-to-month customers churn at ~43%
2. **Tenure** — new customers (< 1 year) are the highest risk group
3. **Fiber optic** internet users churn more (~42%) despite higher cost
4. **Electronic check** payers are more likely to churn than auto-pay customers
5. **Online security & tech support** add-ons significantly reduce churn

---

## 💡 Retention Strategies

- Offer discounts to convert month-to-month → annual contracts
- Target new customers (< 6 months) with loyalty perks
- Bundle Online Security + Tech Support at a discount
- Encourage auto-pay enrollment (bank transfer / credit card)

---

## 🛠️ Technologies

- **Python** 3.9+
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — ML pipeline, models, evaluation
- **Matplotlib / Seaborn** — visualizations
- **Streamlit** — interactive web dashboard
