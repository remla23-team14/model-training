"""Tests to assess dataset features
"""

import numpy as np
import pandas as pd
from scipy.stats import skew


def test_class_balance(dataset):
    # Test the balance of classes (positive and negative reviews)
    class_counts = dataset["Liked"].value_counts()
    positive_count = class_counts[1]
    negative_count = class_counts[0]
    total_count = len(dataset)

    # acceptable threshold for class imbalance
    imbalance_threshold = 0.15

    # Assert that the class distribution is within the acceptable range
    assert abs(positive_count - negative_count) / total_count <= imbalance_threshold


def test_skewness(dataset):
    # Test skewness of the feature distributions
    review_lengths = dataset["Review"].str.len()

    # Calculate the skewness of review lengths
    skewness = skew(review_lengths)

    # Define the acceptable threshold for skewness
    skewness_threshold = 1

    # Assert that the skewness is within the acceptable range
    assert abs(skewness) <= skewness_threshold
