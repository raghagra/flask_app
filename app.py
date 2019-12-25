import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, render_template
import pickle

# app
app = Flask(__name__)
# load model
model = pickle.load(open('model.pkl','rb'))

# routes
@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   

    # convert data into dataframe
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # predictions
    prediction = model.predict(final_features)

    # send back to browser
    output = round(prediction[0],2)

    # return data
    return render_template('index.html', prediction_text='customer survived{}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)