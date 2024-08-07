from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
import joblib

model_path = r'C:\Users\Suptotthita\Downloads\credit_card_model.pkl'
model = joblib.load(model_path)



# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')


# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    features = [float(x) for x in request.form.values()]
    # Convert to numpy array
    input_features = np.array(features).reshape(1, -1)
    # Perform prediction
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        result = 'Fraudulent'
    else:
        result = 'Not Fraudulent'

    return render_template('index.html', prediction_text='Prediction: {}'.format(result))


if __name__ == '__main__':
    app.run(debug=True)
