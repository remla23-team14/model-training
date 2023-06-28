import numpy as np

from src.train import train_pipeline


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
