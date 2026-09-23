# Credit Card Fraud Detection

A Machine Learning classification project that detects whether a credit card transaction is legitimate or fraudulent.

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib

## Dataset

Download the Credit Card Fraud Detection dataset from Kaggle:

https://www.kaggle.com/mlg-ulb/creditcardfraud

After downloading and extracting it, place:

```text
creditcard.csv
```

inside:

```text
data/
└── creditcard.csv
```

The dataset contains transaction features such as `Time`, `Amount`, `V1`–`V28`, and the target column `Class`.

```text
Class = 0 → Legitimate transaction
Class = 1 → Fraudulent transaction
```

## Project Structure

```text
CreditCardFraudDetection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│
├── outputs/
│   └── charts/
│
├── 01_data_cleaning_eda.py
├── 02_train_model.py
├── 03_predict.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Execute the Project

### 1. Data Cleaning and EDA

```powershell
python 01_data_cleaning_eda.py
```

This loads the dataset, checks missing values and duplicates, cleans the data, analyzes class imbalance, and creates Matplotlib charts.

### 2. Train Models

```powershell
python 02_train_model.py
```

The project trains and compares:

* Logistic Regression
* Random Forest

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* PR-AUC
* Confusion Matrix

### 3. Predict a Transaction

```powershell
python 03_predict.py
```

You can test a transaction from the test dataset or provide your own CSV.

The output will indicate:

```text
LEGITIMATE
```

or:

```text
FRAUD
```

along with the predicted fraud probability.

## Important

The dataset is highly imbalanced, with fraud transactions representing a very small fraction of all transactions. Therefore, **Precision, Recall, F1 Score, and PR-AUC are more informative than accuracy alone** for evaluating the fraud detection model.

## Workflow

```text
Kaggle Dataset
      ↓
Data Cleaning
      ↓
EDA + Matplotlib
      ↓
Train/Test Split
      ↓
Preprocessing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Save Model
      ↓
Fraud Prediction
```
