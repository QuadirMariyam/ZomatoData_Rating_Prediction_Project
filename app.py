from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('./model/zomato_model.pkl', 'rb'))
scaler = pickle.load(open('./model/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # collect data from form
    features = [
        request.form['Currency'],
        request.form['table_booking'],
        request.form['online_order'],
        request.form['delivering_now'],
        request.form['price_range'],
        request.form['Votes'],
        request.form['cost_for_two']
    ]
    features = [float(x) for x in features]
    final_features = np.array(features).reshape(1, -1)
    scaled = scaler.transform(final_features)
    prediction = model.predict(scaled)[0]
    
    return render_template('index.html', prediction_text=f'Predicted Rating: {round(prediction,2)}')

if __name__=="__main__":
    app.run(debug=True)