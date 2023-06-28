import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.evaluate import eval_score

np.random.seed(0)


def test_eval_score(model, corpus, dataset, classifier):
    # Define test data
    seed = 2
    X = model.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    # Evaluate the test model
    metrics = eval_score(classifier, X_test, y_test)

    # Extract the metrics
    acc = metrics[0]
    prec = metrics[1]
    rec = metrics[2]
    cm = metrics[3]

    # Check if the calculated metrics match the expected values
    assert acc == accuracy_score(y_test, classifier.predict(X_test))
    assert prec == precision_score(y_test, classifier.predict(X_test))
    assert rec == recall_score(y_test, classifier.predict(X_test))
    assert np.array_equal(cm, confusion_matrix(y_test, classifier.predict(X_test)))
