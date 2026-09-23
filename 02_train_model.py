import os

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "outputs/cleaned_creditcard.csv"
)

MODEL_DIR = "models"

CHART_DIR = "outputs/charts"


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    CHART_DIR,
    exist_ok=True
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 70)
print("CREDIT CARD FRAUD MODEL TRAINING")
print("=" * 70)

df = pd.read_csv(
    DATA_PATH
)


print(
    "\nDataset shape:",
    df.shape
)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=["class"]
)


y = df["class"]


print(
    "\nNumber of features:",
    X.shape[1]
)


print(
    "Number of records:",
    X.shape[0]
)


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


print(
    "\nFraud in training:",
    y_train.sum()
)


print(
    "Fraud in testing:",
    y_test.sum()
)


# ============================================================
# 4. LOGISTIC REGRESSION
# ============================================================

logistic_model = Pipeline(

    steps=[

        (
            "scaler",

            StandardScaler()
        ),

        (
            "model",

            LogisticRegression(

                max_iter=1000,

                class_weight="balanced",

                random_state=42,

                solver="liblinear"
            )
        )
    ]
)


# ============================================================
# 5. RANDOM FOREST
# ============================================================

random_forest_model = Pipeline(

    steps=[

        (
            "model",

            RandomForestClassifier(

                n_estimators=100,

                max_depth=16,

                min_samples_leaf=2,

                class_weight="balanced_subsample",

                random_state=42,

                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"TRAINING {model_name}"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    probabilities = (
        model.predict_proba(X_test)[:, 1]
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )


    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )


    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )


    pr_auc = average_precision_score(
        y_test,
        probabilities
    )


    cm = confusion_matrix(
        y_test,
        predictions
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )


    print(
        f"Precision: {precision:.4f}"
    )


    print(
        f"Recall   : {recall:.4f}"
    )


    print(
        f"F1 Score : {f1:.4f}"
    )


    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )


    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )


    print(
        "\nConfusion Matrix:"
    )


    print(
        cm
    )


    print(
        "\nClassification Report:"
    )


    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Legitimate",
                "Fraud"
            ],
            zero_division=0
        )
    )


    return {

        "name": model_name,

        "model": model,

        "predictions": predictions,

        "probabilities": probabilities,

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "pr_auc": pr_auc,

        "confusion_matrix": cm
    }


# ============================================================
# 7. TRAIN LOGISTIC REGRESSION
# ============================================================

logistic_result = evaluate_model(

    "Logistic Regression",

    logistic_model
)


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

random_forest_result = evaluate_model(

    "Random Forest",

    random_forest_model
)


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame(

    [

        {
            "Model":
                "Logistic Regression",

            "Accuracy":
                logistic_result["accuracy"],

            "Precision":
                logistic_result["precision"],

            "Recall":
                logistic_result["recall"],

            "F1":
                logistic_result["f1"],

            "ROC_AUC":
                logistic_result["roc_auc"],

            "PR_AUC":
                logistic_result["pr_auc"]
        },

        {
            "Model":
                "Random Forest",

            "Accuracy":
                random_forest_result["accuracy"],

            "Precision":
                random_forest_result["precision"],

            "Recall":
                random_forest_result["recall"],

            "F1":
                random_forest_result["f1"],

            "ROC_AUC":
                random_forest_result["roc_auc"],

            "PR_AUC":
                random_forest_result["pr_auc"]
        }
    ]
)


print(
    "\n" + "=" * 70
)


print(
    "MODEL COMPARISON"
)


print(
    "=" * 70
)


print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ============================================================
# 10. SELECT MODEL USING PR-AUC
# ============================================================

if (
    random_forest_result["pr_auc"]
    >=
    logistic_result["pr_auc"]
):

    selected_result = (
        random_forest_result
    )

else:

    selected_result = (
        logistic_result
    )


selected_model = (
    selected_result["model"]
)


selected_name = (
    selected_result["name"]
)


print(
    f"\nSelected model: "
    f"{selected_name}"
)


# ============================================================
# 11. SAVE MODELS
# ============================================================

joblib.dump(

    logistic_result["model"],

    f"{MODEL_DIR}/logistic_fraud_model.pkl"
)


joblib.dump(

    random_forest_result["model"],

    f"{MODEL_DIR}/random_forest_fraud_model.pkl"
)


joblib.dump(

    selected_model,

    f"{MODEL_DIR}/fraud_detection_model.pkl"
)


with open(
    f"{MODEL_DIR}/selected_model.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        selected_name
    )


print(
    "\nModels saved successfully."
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

cm = (
    selected_result["confusion_matrix"]
)


plt.figure(
    figsize=(7, 6)
)


plt.imshow(
    cm
)


plt.colorbar(
    label="Count"
)


plt.xticks(
    [0, 1],
    [
        "Legitimate",
        "Fraud"
    ]
)


plt.yticks(
    [0, 1],
    [
        "Legitimate",
        "Fraud"
    ]
)


plt.xlabel(
    "Predicted"
)


plt.ylabel(
    "Actual"
)


plt.title(
    f"Confusion Matrix - {selected_name}"
)


for row in range(2):

    for col in range(2):

        plt.text(

            col,

            row,

            str(cm[row, col]),

            ha="center",

            va="center"
        )


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/06_confusion_matrix.png",
    dpi=150
)


plt.close()


# ============================================================
# 13. ROC CURVE
# ============================================================

logistic_fpr, logistic_tpr, _ = roc_curve(

    y_test,

    logistic_result["probabilities"]
)


rf_fpr, rf_tpr, _ = roc_curve(

    y_test,

    random_forest_result["probabilities"]
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    logistic_fpr,

    logistic_tpr,

    label=(
        "Logistic Regression "
        f"(AUC = "
        f"{logistic_result['roc_auc']:.3f})"
    )
)


plt.plot(

    rf_fpr,

    rf_tpr,

    label=(
        "Random Forest "
        f"(AUC = "
        f"{random_forest_result['roc_auc']:.3f})"
    )
)


plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    label="Random Classifier"
)


plt.xlabel(
    "False Positive Rate"
)


plt.ylabel(
    "True Positive Rate"
)


plt.title(
    "ROC Curve"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/07_roc_curve.png",
    dpi=150
)


plt.close()


# ============================================================
# 14. PRECISION-RECALL CURVE
# ============================================================

logistic_precision, logistic_recall, _ = (
    precision_recall_curve(
        y_test,
        logistic_result["probabilities"]
    )
)


rf_precision, rf_recall, _ = (
    precision_recall_curve(
        y_test,
        random_forest_result["probabilities"]
    )
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(

    logistic_recall,

    logistic_precision,

    label=(
        "Logistic Regression "
        f"(AP = "
        f"{logistic_result['pr_auc']:.3f})"
    )
)


plt.plot(

    rf_recall,

    rf_precision,

    label=(
        "Random Forest "
        f"(AP = "
        f"{random_forest_result['pr_auc']:.3f})"
    )
)


plt.xlabel(
    "Recall"
)


plt.ylabel(
    "Precision"
)


plt.title(
    "Precision-Recall Curve"
)


plt.legend()


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/08_precision_recall_curve.png",
    dpi=150
)


plt.close()


# ============================================================
# 15. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_pipeline = (
    random_forest_result["model"]
)


rf_model = (
    rf_pipeline.named_steps["model"]
)


feature_importance = pd.DataFrame(

    {

        "feature":

            X.columns,

        "importance":

            rf_model.feature_importances_
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print(
    "\n" + "=" * 70
)


print(
    "TOP 15 FEATURES"
)


print(
    "=" * 70
)


print(
    feature_importance.head(15)
    .to_string(index=False)
)


feature_importance.to_csv(

    "outputs/feature_importance.csv",

    index=False
)


# ============================================================
# 16. FEATURE IMPORTANCE CHART
# ============================================================

top_features = (
    feature_importance
    .head(15)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(10, 7)
)


plt.barh(

    top_features["feature"],

    top_features["importance"]
)


plt.xlabel(
    "Importance"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "Top 15 Random Forest Features"
)


plt.tight_layout()


plt.savefig(
    f"{CHART_DIR}/09_feature_importance.png",
    dpi=150
)


plt.close()


# ============================================================
# 17. SAVE TEST DATA
# ============================================================

test_data = X_test.copy()


test_data["class"] = y_test.values


test_data.to_csv(

    "outputs/test_transactions.csv",

    index=False
)


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 70
)


print(
    "MODEL TRAINING COMPLETED"
)


print(
    "=" * 70
)


print(
    f"\nSelected model: "
    f"{selected_name}"
)


print(
    f"PR-AUC: "
    f"{selected_result['pr_auc']:.4f}"
)


print(
    "\nGenerated files:"
)


print(
    "models/fraud_detection_model.pkl"
)


print(
    "outputs/model_comparison.csv"
)


print(
    "outputs/test_transactions.csv"
)


print(
    "outputs/feature_importance.csv"
)


print(
    "\nGenerated charts:"
)


print(
    "06_confusion_matrix.png"
)


print(
    "07_roc_curve.png"
)


print(
    "08_precision_recall_curve.png"
)


print(
    "09_feature_importance.png"
)