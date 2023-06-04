"""Tests for get_data.py function
"""
import os

import pandas as pd
import pytest

from src.get_data import read_data


@pytest.fixture()
def data_file_path():
    """File path"""
    yield os.path.join("data", "external", "a1_RestaurantReviews_HistoricDump.tsv")


@pytest.fixture()
def output_file_path():
    """Output path"""
    yield os.path.join("data", "raw", "train_dataset.csv")


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
