import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.train import train_pipeline

np.random.seed(0)


def test_non_determinism_robustness(dataset):
    seed_values = [0, 1, 2]  # Seed values to test
    accuracies = []

    for seed in seed_values:
        accuracy = train_pipeline(dataset, seed)
        accuracies.append(accuracy)

    # Check if the accuracies are within a threshold of each other
    threshold = 0.05  # Define the threshold for acceptable accuracy variation
    mean_accuracy = np.mean(accuracies)
    assert all(abs(acc - mean_accuracy) <= threshold for acc in accuracies)


def test_against_baseline(dataset, model, corpus, classifier):
    # Test the Gaussian process classifier against a baseline gaussian NB
    seed = 2

    X = model.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    base_classifier = GaussianNB()
    base_classifier.fit(X_train, y_train)

    y_new = classifier.predict(X_test)
    y_base = base_classifier.predict(X_test)

    acc_new = accuracy_score(y_new, y_test)
    acc_base = accuracy_score(y_base, y_test)

    assert acc_new >= acc_base
