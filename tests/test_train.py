import numpy as np
import pytest

from src.data_preprocessing import _load_data
from src.train import train_pipeline


@pytest.fixture()
def train_dataset():
    # This fixture provides the train dataset for testing
    dataset = _load_data()  # Load or generate the train dataset
    yield dataset


def test_non_determinism_robustness(train_dataset):
    seed_values = [0, 1, 2]  # Seed values to test
    accuracies = []

    for seed in seed_values:
        accuracy = train_pipeline(train_dataset, seed)
        accuracies.append(accuracy)

    # Check if the accuracies are within a threshold of each other
    threshold = 0.05  # Define the threshold for acceptable accuracy variation
    mean_accuracy = np.mean(accuracies)
    assert all(abs(acc - mean_accuracy) <= threshold for acc in accuracies)
