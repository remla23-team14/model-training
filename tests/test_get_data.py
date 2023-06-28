"""Tests for get_data.py function
"""
import pandas as pd

from src.get_data import read_data


def test_data_retrieval(data_file_path):
    dataset = read_data(data_file_path)
    # Check if the dataset is a pandas DataFrame
    assert isinstance(dataset, pd.DataFrame)

    # Check if the dataset has the expected columns
    expected_columns = ["Review", "Liked"]
    assert all(col in dataset.columns for col in expected_columns)

    # Perform other assertions on the dataset as needed


def test_data_dump(output_file_path):
    # Load the dataset from the output file
    dataset = pd.read_csv(output_file_path)

    # Perform assertions to verify the data was dumped correctly
    assert not dataset.empty
