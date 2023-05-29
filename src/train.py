"""Module to perform classifier training"""

import os
import pickle
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier


dataset = pd.read_csv(
    os.path.join('data','raw','train_dataset.csv'),
    dtype={'Review': object, 'Liked': int})
dataset = dataset[["Review","Liked"]]

corpus = joblib.load(os.path.join('data','processed','pre_processed_dataset.joblib'))

with open(os.path.join('data','processed','c1_BoW_Sentiment_Model.pkl'), "rb") as cv_f:
    cv = pickle.load(cv_f)

X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#classifier = GaussianNB()
classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
classifier.fit(X_train, y_train)

output_path = os.path.join('models', 'c2_Classifier_Sentiment_Model')
d_path = os.path.join('data', 'processed', 'XY_data.joblib')
joblib.dump(classifier, output_path)
joblib.dump([X_train, X_test, y_train, y_test], d_path)
#y_pred = classifier.predict(X_test)

#cm = confusion_matrix(y_test, y_pred)
#print(cm)

#acc = accuracy_score(y_test, y_pred)
#f = open('test_acc.txt', "w")
#f.write(str(acc))
#f.close()

#print(accuracy_score(y_test, y_pred))
