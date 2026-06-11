"""
Unit tests for the Census ML pipeline.
"""

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    train_model,
)


DATA_PATH = "data/census.csv"
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
    Clean whitespace from the dataset.
    """
    data = data.copy()
    data.columns = data.columns.str.strip()

    object_columns = data.select_dtypes(include=["object"]).columns

    for column in object_columns:
        data[column] = data[column].str.strip()

    return data


def load_sample_data():
    """
    Load a sample of the Census dataset for testing.
    """
    data = pd.read_csv(DATA_PATH)
    data = clean_data(data)

    sample_size = min(1000, len(data))

    data = data.sample(
        n=sample_size,
        random_state=42,
    )

    return data.reset_index(drop=True)


def test_data_loads_successfully():
    """
    Test that the dataset loads and has required columns.
    """
    data = load_sample_data()

    assert len(data) > 0
    assert LABEL in data.columns

    for feature in CAT_FEATURES:
        assert feature in data.columns


def test_train_model_returns_random_forest():
    """
    Test that train_model returns a RandomForestClassifier.
    """
    data = load_sample_data()

    train, _ = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[LABEL],
    )

    X_train, y_train, _, _ = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_correct_length():
    """
    Test that inference returns one prediction per row.
    """
    data = load_sample_data()

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[LABEL],
    )

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
    preds = inference(model, X_test)

    assert len(preds) == len(y_test)


def test_compute_model_metrics_expected_values():
    """
    Test that metric calculations return expected values.
    """
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 0.5
    assert round(fbeta, 4) == 0.6667


def test_slice_performance_returns_metrics():
    """
    Test that slice performance returns valid metric values.
    """
    data = load_sample_data()

    train, test = train_test_split(
        data,
        test_size=0.20,
        random_state=42,
        stratify=data[LABEL],
    )

    test = test.reset_index(drop=True)

    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )

    model = train_model(X_train, y_train)

    value = test["sex"].iloc[0]

    precision, recall, fbeta = performance_on_categorical_slice(
        test,
        "sex",
        value,
        CAT_FEATURES,
        LABEL,
        encoder,
        lb,
        model,
    )

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0


def test_model_artifact_exists_after_training():
    """
    Test that the training script created a saved model artifact.
    """
    assert os.path.exists("model/model.pkl")
