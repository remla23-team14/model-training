import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle

nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def _load_data():
    dataset = pd.read_csv('a1_RestaurantReviews_HistoricDump.tsv', delimiter = '\t', quoting = 3)
    return dataset

def _process_data(dataset):
   
    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    return corpus

def _transform_data(dataset):
    corpus = _process_data(dataset)
    cv = CountVectorizer(max_features = 1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))
    return X, y

def main():
    dataset = _load_data()
    X, y = _transform_data(dataset)
    

if __name__ == "__main__":
    main()
