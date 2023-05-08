from flask import Flask, send_file
import joblib

app = Flask(__name__)

@app.route("/model", methods=['GET', 'POST'])
def get_model():
    path='c2_Classifier_Sentiment_Model'
    return send_file(path, as_attachment=True)
    #model = joblib.load('c2_Classifier_Sentiment_Model')


if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')