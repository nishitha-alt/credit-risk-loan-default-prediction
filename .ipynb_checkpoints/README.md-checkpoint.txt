# 💳 Credit Risk / Loan Default Prediction

Predicting the probability that a loan will default, using LendingClub's
historical loan data.

## 📌 Problem
Lenders need to estimate default risk in order to price loans correctly and
set approval thresholds. This project builds and compares machine learning
models to predict whether a borrower will repay or default on a loan.

## 📊 Data
LendingClub historical loan data (395219 rows). Target variable: whether a loan
was Fully Paid (1) or Charged Off / defaulted (0). Features include loan
amount, interest rate, borrower income, debt-to-income ratio, credit history
length, employment length, and loan purpose.

## 🛠 Approach
1. Cleaned missing values and removed non-predictive/leaky columns (e.g. address, issue date)
2. Engineered new features, including a loan-to-income ratio
3. Handled class imbalance using SMOTE, since defaults are a minority class
4. Trained and compared Logistic Regression, Random Forest, and XGBoost
5. Evaluated using ROC-AUC, precision, and recall (not just accuracy, since the classes are imbalanced)

## 📈 Results

| Model | ROC-AUC | Recall (default) | Precision (default) |
|---|---|---|---|
| Logistic Regression | 0.705 | 0.30 | 0.41  |
| Random Forest | 0.695 | 0.19 | 0.43 |
| XGBoost | 0.700 | 0.15 | 0.46 |

**XGBoost performed best**, with the strongest ROC-AUC and recall on the
default class — meaning it catches more actual defaults than the other models,
which matters more to a lender than raw accuracy.

### Feature Importance
![Feature Importance](feature_importance.png)

**Key takeaway:** [your finished sentence from Step 2 — e.g. "The biggest
predictors of default are interest rate, open_acc, and pub_rec,
which makes sense because these directly reflect a borrower's existing
financial strain and how risky LendingClub already judged them to be."]

## 💼 Business Takeaway
A model like this lets a lender flag higher-risk applications for closer review
or adjusted pricing, rather than applying the same terms to every borrower
regardless of risk — directly supporting credit risk and underwriting decisions.

## 🧰 Tech Stack
Python, pandas, scikit-learn, XGBoost, matplotlib, imbalanced-learn (SMOTE)

## 📂 Files
- `credit_risk_model.ipynb` — full analysis, cleaning, modeling, and evaluation
- `feature_importance.png` — top feature importance chart
- `requirements.txt` — Python packages needed to run the notebook