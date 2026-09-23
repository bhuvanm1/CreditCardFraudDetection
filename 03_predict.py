from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(
    "models/fraud_detection_model.pkl"
)

TEST_DATA_PATH = Path(
    "outputs/test_transactions.csv"
)


# ============================================================
# CHECK MODEL
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        "\nModel not found.\n"
        "Run this first:\n\n"
        "python 02_train_model.py\n"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# EXPECTED FEATURES
# ============================================================

EXPECTED_FEATURES = [
    "time"
]

EXPECTED_FEATURES += [
    f"v{i}"
    for i in range(1, 29)
]

EXPECTED_FEATURES += [
    "amount"
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def predict_transactions(
    transactions
):

    # Make sure required columns exist
    missing_columns = [

        column

        for column in EXPECTED_FEATURES

        if column not in transactions.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing columns: "
            + ", ".join(missing_columns)
        )


    # Keep only model features
    X_new = transactions[
        EXPECTED_FEATURES
    ]


    # Prediction
    predictions = model.predict(
        X_new
    )


    # Fraud probabilities
    probabilities = (
        model.predict_proba(
            X_new
        )[:, 1]
    )


    return (
        predictions,
        probabilities
    )


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 70)
print("CREDIT CARD FRAUD PREDICTION")
print("=" * 70)


print(
    "\nChoose an option:"
)


print(
    "\n1. Test a transaction from the held-out test dataset"
)


print(
    "2. Predict transactions from your own CSV"
)


choice = input(
    "\nEnter 1 or 2: "
).strip()


# ============================================================
# OPTION 1
# ============================================================

if choice == "1":

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            "\nTest transaction file not found.\n"
            "Run 02_train_model.py first."
        )


    test_data = pd.read_csv(
        TEST_DATA_PATH
    )


    print(
        f"\nNumber of test transactions: "
        f"{len(test_data)}"
    )


    while True:

        try:

            index = int(
                input(
                    f"\nEnter transaction number "
                    f"(0 to {len(test_data) - 1}): "
                )
            )


            if (
                0 <= index < len(test_data)
            ):

                break


            print(
                "Invalid transaction number."
            )


        except ValueError:

            print(
                "Please enter a whole number."
            )


    selected_transaction = (
        test_data.iloc[[index]]
        .copy()
    )


    actual_class = int(
        selected_transaction["class"].iloc[0]
    )


    transaction_features = (
        selected_transaction[
            EXPECTED_FEATURES
        ]
    )


    predictions, probabilities = (
        predict_transactions(
            transaction_features
        )
    )


    prediction = int(
        predictions[0]
    )


    fraud_probability = (
        probabilities[0] * 100
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )


    print(
        "TRANSACTION RESULT"
    )


    print(
        "=" * 70
    )


    print(
        f"\nActual class: "
        f"{'FRAUD' if actual_class == 1 else 'LEGITIMATE'}"
    )


    print(
        f"Predicted class: "
        f"{'FRAUD' if prediction == 1 else 'LEGITIMATE'}"
    )


    print(
        f"Fraud probability: "
        f"{fraud_probability:.2f}%"
    )


    if prediction == 1:

        print(
            "\nModel decision: "
            "POTENTIAL FRAUD"
        )

    else:

        print(
            "\nModel decision: "
            "LIKELY LEGITIMATE"
        )


# ============================================================
# OPTION 2
# ============================================================

elif choice == "2":

    csv_path = input(
        "\nEnter path to your transaction CSV: "
    ).strip()


    custom_path = Path(
        csv_path
    )


    if not custom_path.exists():

        raise FileNotFoundError(
            f"\nFile not found: "
            f"{custom_path}"
        )


    custom_data = pd.read_csv(
        custom_path
    )


    predictions, probabilities = (
        predict_transactions(
            custom_data
        )
    )


    results = custom_data.copy()


    results["fraud_prediction"] = (
        predictions
    )


    results["fraud_probability"] = (
        probabilities
    )


    results["fraud_prediction"] = (
        results["fraud_prediction"]
        .map({
            0: "LEGITIMATE",
            1: "FRAUD"
        })
    )


    output_path = (
        "outputs/predictions.csv"
    )


    results.to_csv(
        output_path,
        index=False
    )


    print(
        "\nPredictions saved to:"
    )


    print(
        output_path
    )


    print(
        "\nPrediction summary:"
    )


    print(
        results[
            [
                "fraud_prediction",
                "fraud_probability"
            ]
        ]
    )


else:

    print(
        "\nInvalid option."
    )

    print(
        "Please run the program again "
        "and choose 1 or 2."
    )