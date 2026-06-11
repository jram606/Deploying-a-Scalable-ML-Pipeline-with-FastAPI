"""
Train the Census income classification model.
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


PROJECT_PATH = os.getcwd()
DATA_PATH = os.path.join(PROJECT_PATH, "data", "census.csv")
MODEL_PATH = os.path.join(PROJECT_PATH, "model", "model.pkl")
ENCODER_PATH = os.path.join(PROJECT_PATH, "model", "encoder.pkl")
SLICE_OUTPUT_PATH = os.path.join(PROJECT_PATH, "slice_output.txt")
LABEL = "salary"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def clean_data(data):
    """
    Remove extra whitespace from column names and string values.
    """
    data = data.copy()
    data.columns = data.columns.str.strip()

    object_columns = data.select_dtypes(include=["object"]).columns

    for column in object_columns:
        data[column] = data[column].str.strip()

    return data


def main():
    """
    Run the full training pipeline.
    """
    os.makedirs("model", exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    data = clean_data(data)

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[LABEL],
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    save_model(model, MODEL_PATH)
    save_model(encoder, ENCODER_PATH)

    model = load_model(MODEL_PATH)

    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | F1: {fbeta:.4f}"
    )

    with open(SLICE_OUTPUT_PATH, "w", encoding="utf-8") as file:
        for column in CAT_FEATURES:
            for slice_value in sorted(test[column].unique()):
                count = test[test[column] == slice_value].shape[0]

                p_slice, r_slice, fb_slice = (
                    performance_on_categorical_slice(
                        test,
                        column,
                        slice_value,
                        CAT_FEATURES,
                        LABEL,
                        encoder,
                        lb,
                        model,
                    )
                )

                print(
                    f"{column}: {slice_value}, Count: {count:,}",
                    file=file,
                )
                print(
                    f"Precision: {p_slice:.4f} | "
                    f"Recall: {r_slice:.4f} | F1: {fb_slice:.4f}",
                    file=file,
                )

    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")
    print(f"Slice output saved to {SLICE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
