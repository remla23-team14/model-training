# pylint: skip-file

import os

import joblib
import libpy
import pytest

from src.data_preprocessing import train_dataset_path


@pytest.fixture()
def data_file_path():
    """File path"""
    yield os.path.join("data", "external", "a1_RestaurantReviews_HistoricDump.tsv")


@pytest.fixture()
def output_file_path():
    """Output path"""
    yield os.path.join("data", "raw", "train_dataset.csv")


@pytest.fixture()
def dataset():
    # This fixture provides the train dataset for testing
    dataset = libpy.load_data(train_dataset_path)  # Load or generate the train dataset
    yield dataset


@pytest.fixture()
def corpus():
    # This fixture provides the corpus for testing
    corpus_path = os.path.join("data", "processed", "pre_processed_dataset.joblib")
    corpus = joblib.load(corpus_path)  # Load or generate the train dataset
    yield corpus


@pytest.fixture()
def model():
    # This fixture provides the model for testing
    model_path = os.path.join("data", "processed", "c1_BoW_Sentiment_Model.joblib")
    model = joblib.load(model_path)  # Load or generate the train dataset
    yield model


@pytest.fixture()
def classifier():
    # This fixture provides the trained classifier for testing
    classifier_path = os.path.join("models", "c2_Classifier_Sentiment_Model")
    classifier = joblib.load(classifier_path)  # Load or generate the train dataset
    yield classifier
