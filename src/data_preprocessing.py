"""
Module to preprocess the train dataset and store it in a file"""

import os
import re
from typing import List

import joblib  # type: ignore
import nltk  # type: ignore
import pandas as pd
from nltk.corpus import stopwords  # type: ignore
from nltk.stem.porter import PorterStemmer  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore

nltk.download("stopwords")


def _load_data() -> pd.DataFrame:
    """Function to load data from a csv file"""
    dataset = pd.read_csv(
        os.path.join("data", "raw", "train_dataset.csv"),
        dtype={"Review": object, "Liked": int},
    )
    dataset = dataset[["Review", "Liked"]]
    return dataset


def _process_data(dataset: pd.DataFrame) -> List[str]:
    """Function to perform some basic preprocessing on the loaded dataset"""
    ps_obj = PorterStemmer()

    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")

    corpus = []

    for i in range(0, 900):
        review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
        review = review.lower()
        review = review.split()
        review = [
            ps_obj.stem(word) for word in review if not word in set(all_stopwords)
        ]
        review = " ".join(review)
        corpus.append(review)

    return corpus


def main() -> None:
    """Preprocess dataset and write it to file"""
    dataset = _load_data()
    corpus = _process_data(dataset)
    cvec = CountVectorizer(max_features=1420)
    bow_path = os.path.join("data", "processed", "c1_BoW_Sentiment_Model.joblib")
    corpus_path = os.path.join("data", "processed", "pre_processed_dataset.joblib")

    joblib.dump(cvec, bow_path)
    joblib.dump(corpus, corpus_path)


if __name__ == "__main__":
    main()
