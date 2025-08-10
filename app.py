# Import necessary libraries
from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
import pandas as pd

# Initialize the Flask application
app = Flask(__name__)

# --- Load the Model and Scaler ---
try:
    with open('pmsm_random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('pmsm_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: Model or scaler files not found. Make sure they are in the correct directory.")
    model = None
    scaler = None

# --- Define the routes ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text='Error: Model not loaded.')

    try:
        # --- Robustly get data from the form by name ---
        # This is better than relying on order.
        input_features = {
            'ambient': float(request.form.get('ambient')),
            'coolant': float(request.form.get('coolant')),
            'u_d': float(request.form.get('u_d')),
            'u_q': float(request.form.get('u_q')),
            'motor_speed': float(request.form.get('motor_speed')),
            'torque': float(request.form.get('torque')),
            'i_d': float(request.form.get('i_d')),
            'i_q': float(request.form.get('i_q')),
            'stator_yoke': float(request.form.get('stator_yoke')),
            'stator_tooth': float(request.form.get('stator_tooth')),
            'stator_winding': float(request.form.get('stator_winding')),
        }

        # --- Corrected Feature Engineering ---
        # Use **2 for squaring, not *2
        input_features['i_mag_sq'] = input_features['i_d']**2 + input_features['i_q']**2
        input_features['p_elec'] = input_features['u_d'] * input_features['i_d'] + input_features['u_q'] * input_features['i_q']
        input_features['temp_diff_stator_coolant'] = input_features['stator_yoke'] - input_features['coolant']

        # --- Prepare data for prediction ---
        # Get the exact column order from your notebook.
        # This list MUST match the one your model was trained on.
        training_columns = ['u_q', 'coolant', 'stator_winding', 'u_d', 'stator_tooth', 'motor_speed', 'i_d', 'i_q', 'stator_yoke', 'ambient', 'torque', 'i_mag_sq', 'p_elec', 'temp_diff_stator_coolant']
        
        # Create a DataFrame from the input data with the correct column order
        final_features = pd.DataFrame([input_features], columns=training_columns)

        # Make the prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        # Render the page again with the prediction
        return render_template('index.html', prediction_text=f'Estimated Rotor Temperature: {output} °C')

    except Exception as e:
        # Show a more helpful error message
        return render_template('index.html', prediction_text=f'Error: {e}')

# This allows you to run the app directly from the command line
if __name__ == "__main__":
    app.run(debug=True)
