"""Flask endpoint to download trained model"""
import os
from flask import Flask, send_file

app = Flask(__name__)

@app.route("/model", methods=['GET', 'POST'])
def get_model():
    """ Fetch and make the trained model available"""
    path=os.path.join('models', 'c2_Classifier_Sentiment_Model')
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')
