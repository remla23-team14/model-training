"""
Module to preprocess the train dataset and store it in a file"""

import os
import pathlib

import joblib  # type: ignore
import nltk  # type: ignore
import libpy
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

nltk.download("stopwords")


train_dataset_path = pathlib.Path('data') / 'raw' / 'train_dataset.csv'


def main() -> None:
    """Preprocess dataset and write it to file"""
    dataset = libpy.load_data(train_dataset_path)
    corpus = libpy.process_data(dataset)
    cvec = CountVectorizer(max_features=1420)
    bow_path = os.path.join("data", "processed", "c1_BoW_Sentiment_Model.joblib")
    corpus_path = os.path.join("data", "processed", "pre_processed_dataset.joblib")

    joblib.dump(cvec, bow_path)
    joblib.dump(corpus, corpus_path)


if __name__ == "__main__":
    main()
