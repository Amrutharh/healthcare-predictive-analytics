from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Valid ranges for input validation
VALID_RANGES = {
    'Pregnancies': (0, 20),
    'Glucose': (30, 300),
    'BloodPressure': (20, 200),
    'SkinThickness': (0, 80),
    'Insulin': (0, 900),
    'BMI': (5, 70),
    'DiabetesPedigreeFunction': (0, 2.5),
    'Age': (1, 120)
}

# Fields where 0 is not medically valid (except Pregnancies)
NON_ZERO_FIELDS = ['Glucose', 'BloodPressure', 'BMI', 'Age']

def validate_input(data):
    """Validate input data and return list of errors"""
    errors = []
    
    # Check if all values are 0 (timepass)
    all_zero = all(v == 0 for v in data)
    if all_zero:
        return ["All values cannot be zero. Please enter real patient data."]
    
    # Check each field
    for i, feature in enumerate(FEATURES):
        value = data[i]
        min_val, max_val = VALID_RANGES[feature]
        
        # Check negative values
        if value < 0:
            errors.append(f"{feature} cannot be negative")
        
        # Check if value is 0 but shouldn't be
        if feature in NON_ZERO_FIELDS and value == 0:
            errors.append(f"{feature} cannot be zero (medically impossible)")
        
        # Check range
        if value < min_val and value != 0:
            errors.append(f"{feature} is too low (minimum: {min_val})")
        elif value > max_val:
            errors.append(f"{feature} is too high (maximum: {max_val})")
    
    return errors

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(request.form[f]) for f in FEATURES]
        
        # Validate input
        errors = validate_input(data)
        if errors:
            return render_template('index.html', validation_errors=errors, values=dict(zip(FEATURES, data)))
        
        # Scale and predict
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'confidence': float(max(probability)) * 100,
            'probability_diabetic': float(probability[1]) * 100,
            'probability_not_diabetic': float(probability[0]) * 100
        }
        
        return render_template('index.html', result=result, values=dict(zip(FEATURES, data)))
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.json
        features = [data[f] for f in FEATURES]
        
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'label': 'Diabetic' if prediction == 1 else 'Not Diabetic',
            'confidence': float(max(probability)) * 100,
            'probability_diabetic': float(probability[1]) * 100,
            'probability_not_diabetic': float(probability[0]) * 100
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
