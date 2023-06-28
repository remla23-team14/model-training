"""Module to perform classifier training"""

import os
from typing import Any

import joblib  # type: ignore
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier  # type: ignore

# from sklearn.naive_bayes import GaussianNB # type: ignore
from sklearn.gaussian_process.kernels import RBF  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

np.random.seed(0)

train_dataset = pd.read_csv(
    os.path.join("data", "raw", "train_dataset.csv"),
    dtype={"Review": object, "Liked": int},
)
train_dataset = train_dataset[["Review", "Liked"]]


def train_pipeline(dataset: pd.DataFrame, seed: int) -> Any:
    """Function to run model training"""

    corpus = joblib.load(
        os.path.join("data", "processed", "pre_processed_dataset.joblib")
    )
    train_cv = joblib.load(
        os.path.join("data", "processed", "c1_BoW_Sentiment_Model.joblib")
    )

    X = train_cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    # classifier = GaussianNB()
    classifier = GaussianProcessClassifier(1.0 * RBF(1.0), random_state=seed)
    classifier.fit(X_train, y_train)

    output_path = os.path.join("models", "c2_Classifier_Sentiment_Model")
    d_path = os.path.join("data", "processed", "XY_data.joblib")
    joblib.dump(classifier, output_path)
    joblib.dump([X_train, X_test, y_train, y_test], d_path)

    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    return acc


if __name__ == "__main__":
    train_acc = train_pipeline(train_dataset, seed=2)

# y_pred = classifier.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# acc = accuracy_score(y_test, y_pred)
# f = open('test_acc.txt', "w")
# f.write(str(acc))
# f.close()

# print(accuracy_score(y_test, y_pred))
