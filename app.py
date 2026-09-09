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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(request.form[f]) for f in FEATURES]
        
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
