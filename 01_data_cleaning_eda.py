import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

RAW_PATH = "data/creditcard.csv"

OUTPUT_PATH = "outputs/cleaned_creditcard.csv"

CHART_DIR = "outputs/charts"


os.makedirs("outputs", exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("CREDIT CARD FRAUD DETECTION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(RAW_PATH)

print("Dataset loaded successfully.")


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("RAW DATASET INFORMATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

print("\nCleaned columns:")
print(df.columns.tolist())


# ============================================================
# 4. CONVERT ALL COLUMNS TO NUMERIC
# ============================================================

for column in df.columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

print("\nMissing values after numeric conversion:")

print(
    df.isnull().sum()
)


# Drop rows where target is missing
df = df.dropna(
    subset=["class"]
).copy()


# Fill missing feature values with median
feature_columns = [
    column
    for column in df.columns
    if column != "class"
]


for column in feature_columns:

    if df[column].isnull().any():

        median_value = (
            df[column].median()
        )

        df[column] = df[column].fillna(
            median_value
        )


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

duplicates_before = (
    df.duplicated().sum()
)

print(
    "\nDuplicates before removal:",
    duplicates_before
)


df = df.drop_duplicates().copy()


duplicates_after = (
    df.duplicated().sum()
)

print(
    "Duplicates after removal:",
    duplicates_after
)


# ============================================================
# 7. CHECK TARGET VALUES
# ============================================================

print("\nTarget values:")

print(
    df["class"].value_counts()
)


# Keep only valid target values
df = df[
    df["class"].isin([0, 1])
].copy()


# Convert target to integer
df["class"] = df["class"].astype(int)


# ============================================================
# 8. FRAUD STATISTICS
# ============================================================

total_transactions = len(df)

fraud_transactions = (
    df["class"] == 1
).sum()

legitimate_transactions = (
    df["class"] == 0
).sum()

fraud_percentage = (
    fraud_transactions
    / total_transactions
    * 100
)


print("\n" + "=" * 70)
print("FRAUD STATISTICS")
print("=" * 70)

print(
    f"\nTotal transactions: "
    f"{total_transactions:,}"
)

print(
    f"Legitimate transactions: "
    f"{legitimate_transactions:,}"
)

print(
    f"Fraudulent transactions: "
    f"{fraud_transactions:,}"
)

print(
    f"Fraud percentage: "
    f"{fraud_percentage:.4f}%"
)


# ============================================================
# 9. SAVE CLEANED DATASET
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    f"\nCleaned dataset saved to:"
    f"\n{OUTPUT_PATH}"
)


# ============================================================
#                    EDA CHARTS
# ============================================================


# ============================================================
# 10. FRAUD DISTRIBUTION
# ============================================================

class_counts = (
    df["class"]
    .value_counts()
    .sort_index()
)


labels = [
    "Legitimate",
    "Fraud"
]


plt.figure(
    figsize=(7, 5)
)


plt.bar(
    labels,
    class_counts.values
)


plt.xlabel(
    "Transaction Type"
)

plt.ylabel(
    "Number of Transactions"
)

plt.title(
    "Legitimate vs Fraudulent Transactions"
)


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/01_class_distribution.png",
    dpi=150
)


plt.close()


# ============================================================
# 11. TRANSACTION AMOUNT DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.hist(
    np.log1p(df["amount"]),
    bins=50
)


plt.xlabel(
    "Log(1 + Transaction Amount)"
)

plt.ylabel(
    "Number of Transactions"
)

plt.title(
    "Transaction Amount Distribution"
)


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/02_amount_distribution.png",
    dpi=150
)


plt.close()


# ============================================================
# 12. AMOUNT BY CLASS
# ============================================================

legitimate_amounts = df.loc[
    df["class"] == 0,
    "amount"
]


fraud_amounts = df.loc[
    df["class"] == 1,
    "amount"
]


plt.figure(
    figsize=(8, 5)
)


plt.boxplot(
    [
        legitimate_amounts,
        fraud_amounts
    ],
    tick_labels=[
        "Legitimate",
        "Fraud"
    ]
)


plt.xlabel(
    "Transaction Type"
)

plt.ylabel(
    "Transaction Amount"
)

plt.title(
    "Transaction Amount by Class"
)


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/03_amount_by_class.png",
    dpi=150
)


plt.close()


# ============================================================
# 13. TIME DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)


plt.hist(
    df.loc[
        df["class"] == 0,
        "time"
    ],
    bins=50,
    alpha=0.6,
    label="Legitimate"
)


plt.hist(
    df.loc[
        df["class"] == 1,
        "time"
    ],
    bins=50,
    alpha=0.6,
    label="Fraud"
)


plt.xlabel(
    "Time"
)

plt.ylabel(
    "Number of Transactions"
)

plt.title(
    "Transaction Time Distribution"
)

plt.legend()


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/04_time_distribution.png",
    dpi=150
)


plt.close()


# ============================================================
# 14. CORRELATION MATRIX
# ============================================================

correlation = df.corr()


plt.figure(
    figsize=(12, 10)
)


plt.imshow(
    correlation,
    aspect="auto"
)


plt.colorbar(
    label="Correlation"
)


plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90,
    fontsize=6
)


plt.yticks(
    range(len(correlation.columns)),
    correlation.columns,
    fontsize=6
)


plt.title(
    "Correlation Matrix"
)


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/05_correlation_matrix.png",
    dpi=150
)


plt.close()


# ============================================================
# 15. FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING AND EDA COMPLETED")
print("=" * 70)

print(
    "\nCharts created in:"
)

print(
    CHART_DIR
)