"""Evaluates the trained classifier and stores the metrics"""

import json
import os

import joblib  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore

# Load data and model
c_path = os.path.join("models", "c2_Classifier_Sentiment_Model")
d_path = os.path.join("data", "processed", "XY_data.joblib")
[X_train, X_test, y_train, y_test] = joblib.load(d_path)
classifier = joblib.load(c_path)

# Model Validation and metrics

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)

print(cm)
print(accuracy_score)

acc_dict = {"accuracy": accuracy_score}
out_path = os.path.join("data", "output", "model_accuracy.json")
with open(out_path, "w", encoding="utf8") as f_out:
    json.dump(acc_dict, f_out)
