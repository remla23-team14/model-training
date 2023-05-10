# model-training


1. Visualizing the data

```
$ python3 src/read_data.py
```
2. Install dependencies

```
pip install -r requirements.txt
```

3. Running the model training & inference steps

```
$ python3 src/train.py
```

The trained model is stored as c2_Classifier_Sentiment_Model and the BoW dictionary (used in preprocessing data during inference) is stored as c1_BoW_Sentiment_Model.pkl 

Training results:

accuracy: 72.77%

Confusion matrix: 
[[67 11]
 [38 64]]
