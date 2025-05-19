import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from pathlib import Path
import os

app = Flask(__name__)

# Load model (same as before)
base_dir = Path(__file__).parent
model_path = base_dir / "app" / "utils" / "stage_prediction_model.pkl"  # Added "app" level
scaler_path = base_dir / "app" / "utils" / "scaler.pkl"  # Added "app" level
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    app.logger.error(f"Initialization error: {str(e)}")
    raise e

# Your prediction function and route handlers remain the same
def predict_treated_water(raw_turbidity, raw_ph, raw_conductivity):
    input_data = pd.DataFrame([[raw_turbidity, raw_ph, raw_conductivity]],
                            columns=['Raw_Water_Turbidity', 'Raw_Water_PH', 'Raw_Water_Conductivity'])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        raw_turbidity = data.get('raw_turbidity')
        raw_ph = data.get('raw_ph')
        raw_conductivity = data.get('raw_conductivity')

        if None in [raw_turbidity, raw_ph, raw_conductivity]:
            return jsonify({'error': 'Missing input values'}), 400

        prediction = predict_treated_water(raw_turbidity, raw_ph, raw_conductivity)
        prediction_list = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction

        if len(prediction_list) != 3:
            return jsonify({'error': f'Expected 3 values, got {len(prediction_list)}'}), 500

        return jsonify({
            'treated_turbidity': float(prediction_list[0]),
            'treated_ph': float(prediction_list[1]),
            'treated_conductivity': float(prediction_list[2])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)