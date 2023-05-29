"""
Module to preprocess the train dataset and store it in a file"""

import os
import re
import pickle
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')


def _load_data():
    """Function to load data from a csv file"""
    dataset = pd.read_csv(
        os.path.join('data','raw','train_dataset.csv'),
        dtype={'Review': object, 'Liked': int})
    dataset = dataset[["Review","Liked"]]
    return dataset

def _process_data(dataset):
    """Function to perform some basic preprocessing on the loaded dataset"""
    ps_obj = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps_obj.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus


def main():
    """Preprocess dataset and write it to file"""
    dataset = _load_data()
    corpus = _process_data(dataset)
    cvec = CountVectorizer(max_features = 1420)
    bow_path = os.path.join('data','processed','c1_BoW_Sentiment_Model.pkl')
    corpus_path = os.path.join('data','processed','pre_processed_dataset.joblib')

    with open(bow_path, "wb") as f:
        pickle.dump(cvec, f)

    joblib.dump(corpus, corpus_path)


if __name__ == "__main__":
    main()
