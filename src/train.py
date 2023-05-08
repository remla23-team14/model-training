import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score
import data_preprocessing

dataset = data_preprocessing._load_data()
X, y = data_preprocessing._transform_data(dataset)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
classifier = GaussianNB()
classifier.fit(X_train, y_train)
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model') 
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))