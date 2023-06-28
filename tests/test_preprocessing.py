import os

import joblib
import pandas as pd
from libpy import load_data
from sklearn.feature_extraction.text import CountVectorizer

from src import data_preprocessing


def test_load_data():
    # Define the path to the test dataset
    test_dataset_path = os.path.join("data", "test", "test_dataset.csv")

    # Create a test dataset
    test_dataset = pd.DataFrame(
        {
            "Review": ["This is a positive review", "This is a negative review"],
            "Liked": [1, 0],
        }
    )

    # Save the test dataset as a CSV file
    test_dataset.to_csv(test_dataset_path, index=False)

    # Load the test dataset
    loaded_data = load_data(test_dataset_path)

    # Check if the loaded data matches the original test dataset
    assert loaded_data.equals(test_dataset)

    # Remove the test dataset file
    os.remove(test_dataset_path)


def test_main():
    # Define the test dataset path
    test_dataset_path = os.path.join("data", "test", "test_dataset.csv")

    # Create a test dataset
    test_dataset = pd.DataFrame(
        {
            "Review": ["This is a positive review", "This is a negative review"],
            "Liked": [1, 0],
        }
    )

    # Save the test dataset as a CSV file
    test_dataset.to_csv(test_dataset_path, index=False)

    # Run the main preprocessing function
    data_preprocessing.main()

    # Define the expected file paths for the preprocessed data
    expected_bow_path = os.path.join(
        "data", "processed", "c1_BoW_Sentiment_Model.joblib"
    )
    expected_corpus_path = os.path.join(
        "data", "processed", "pre_processed_dataset.joblib"
    )

    # Check if the preprocessed data files exist
    assert os.path.exists(expected_bow_path)
    assert os.path.exists(expected_corpus_path)

    # Load the preprocessed data
    cvec = joblib.load(expected_bow_path)
    corpus = joblib.load(expected_corpus_path)

    # Check if the loaded data matches the expected preprocessed data
    assert isinstance(cvec, CountVectorizer)
    assert isinstance(corpus, list)

    # Remove the test dataset and preprocessed data files
    os.remove(test_dataset_path)
