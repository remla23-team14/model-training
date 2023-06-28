"""Evaluates the trained classifier and stores the metrics"""

import json
import os
from typing import Any, List

import joblib  # type: ignore
import numpy.typing as npt
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline  # type: ignore

# Load data and model
c_path = os.path.join("models", "c2_Classifier_Sentiment_Model")
d_path = os.path.join("data", "processed", "XY_data.joblib")
[X_train, X_test, y_train, y_test] = joblib.load(d_path)
classifier = joblib.load(c_path)

# Model Validation and metrics


def eval_score(model: Pipeline, X: npt.ArrayLike, Y: npt.ArrayLike) -> List[Any]:
    """Evaluate metrics"""
    y_pred = model.predict(X)
    acc = accuracy_score(Y, y_pred)
    prec = precision_score(Y, y_pred)
    rec = recall_score(Y, y_pred)
    cm = confusion_matrix(Y, y_pred)

    metrics = [acc, prec, rec, cm]
    return metrics


train_metrics = eval_score(classifier, X_train, y_train)
test_metrics = eval_score(classifier, X_test, y_test)


train_accuracy_score = train_metrics[0]
test_accuracy_score = test_metrics[0]

train_precision_score = train_metrics[1]
test_precision_score = test_metrics[1]

train_recall_score = train_metrics[2]
test_recall_score = test_metrics[2]

print(test_metrics[3])
# print(accuracy_score)

acc_dict = {
    "accuracy": {"train": train_accuracy_score, "test": test_accuracy_score},
    "precision": {"train": train_precision_score, "test": test_precision_score},
    "recall": {"train": train_recall_score, "test": test_recall_score},
}
out_path = os.path.join("data", "output", "model_metrics.json")
with open(out_path, "w", encoding="utf8") as f_out:
    json.dump(acc_dict, f_out)
