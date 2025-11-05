from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load your trained model
model = load('nifty50model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        open_ = float(request.form['Open'])
        high = float(request.form['High'])
        low = float(request.form['Low'])
        last = float(request.form['Last'])
        prev_close = float(request.form['Prev_Close'])
        vwap = float(request.form['VWAP'])

        # Arrange in same order as training
        features = np.array([[open_, high, low, last, prev_close, vwap]])

        # Predict using model
        prediction = model.predict(features)[0]

        # Return formatted result
        return render_template('index.html',
                               prediction_text=f'Predicted Nifty 50 Close: {float(prediction):.2f}'
                               )
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {e}'
                               )

if __name__ == '__main__':
    app.run(debug=True)
