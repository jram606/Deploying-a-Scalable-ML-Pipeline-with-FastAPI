"""
Model training, inference, metrics, and persistence utilities.
"""

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


def train_model(X_train, y_train):
    """
    Train a machine learning model and return it.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Calculate precision, recall, and F1.
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inference and return predictions.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """
    Save a model, encoder, or label binarizer to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    """
    Load a saved model, encoder, or label binarizer from disk.
    """
    with open(path, "rb") as file:
        model = pickle.load(file)

    return model


def performance_on_categorical_slice(
    data,
    column_name,
    slice_value,
    categorical_features,
    label,
    encoder,
    lb,
    model,
):
    """
    Compute model metrics for a specific categorical slice.
    """
    data_slice = data[data[column_name] == slice_value]

    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)

    return precision, recall, fbeta
