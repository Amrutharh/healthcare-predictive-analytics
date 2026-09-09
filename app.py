from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import os
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names
FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Feature display names
FEATURE_NAMES = {
    'Pregnancies': 'Pregnancies',
    'Glucose': 'Glucose (mg/dL)',
    'BloodPressure': 'Blood Pressure (mm Hg)',
    'SkinThickness': 'Skin Thickness (mm)',
    'Insulin': 'Insulin (mu U/ml)',
    'BMI': 'BMI',
    'DiabetesPedigreeFunction': 'Diabetes Pedigree',
    'Age': 'Age (years)'
}

# Healthy ranges for risk analysis
HEALTHY_RANGES = {
    'Glucose': (70, 140, 'mg/dL'),
    'BloodPressure': (60, 80, 'mm Hg'),
    'BMI': (18.5, 24.9, ''),
    'Insulin': (16, 166, 'mu U/ml'),
    'SkinThickness': (10, 50, 'mm'),
    'Age': (18, 45, 'years'),
    'DiabetesPedigreeFunction': (0, 0.5, ''),
    'Pregnancies': (0, 6, '')
}

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

# Feature importance from Random Forest (approximate values)
FEATURE_IMPORTANCE = {
    'Glucose': 0.28,
    'BMI': 0.18,
    'Age': 0.15,
    'DiabetesPedigreeFunction': 0.12,
    'Insulin': 0.10,
    'BloodPressure': 0.08,
    'Pregnancies': 0.05,
    'SkinThickness': 0.04
}

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

def analyze_risk(data):
    """Analyze risk factors and return detailed breakdown"""
    risk_factors = []
    
    for i, feature in enumerate(FEATURES):
        value = data[i]
        
        if feature in HEALTHY_RANGES:
            low, high, unit = HEALTHY_RANGES[feature]
            
            if value == 0 and feature in NON_ZERO_FIELDS:
                status = 'invalid'
                level = 'High Risk'
                color = '#e74c3c'
                message = 'Cannot be zero'
            elif value < low:
                status = 'low'
                level = 'Low' if feature != 'Glucose' else 'Concern'
                color = '#f39c12' if feature == 'Glucose' else '#3498db'
                message = f'Below normal ({low}-{high} {unit})'
            elif value > high:
                if value > high * 1.5:
                    status = 'very_high'
                    level = 'High Risk'
                    color = '#e74c3c'
                    message = f'Significantly above normal ({low}-{high} {unit})'
                else:
                    status = 'high'
                    level = 'Moderate Risk'
                    color = '#e67e22'
                    message = f'Above normal ({low}-{high} {unit})'
            else:
                status = 'normal'
                level = 'Normal'
                color = '#27ae60'
                message = f'Within healthy range ({low}-{high} {unit})'
            
            risk_factors.append({
                'feature': feature,
                'display_name': FEATURE_NAMES[feature],
                'value': value,
                'unit': unit,
                'status': status,
                'level': level,
                'color': color,
                'message': message,
                'healthy_range': f'{low}-{high} {unit}'
            })
    
    # Sort by risk level (high risk first)
    risk_order = {'invalid': 0, 'very_high': 1, 'high': 2, 'low': 3, 'normal': 4}
    risk_factors.sort(key=lambda x: risk_order.get(x['status'], 5))
    
    return risk_factors

def generate_contribution_chart(data, prediction):
    """Generate feature contribution chart using matplotlib"""
    contributions = []
    
    for i, feature in enumerate(FEATURES):
        value = data[i]
        importance = FEATURE_IMPORTANCE[feature]
        
        if feature in HEALTHY_RANGES:
            low, high, _ = HEALTHY_RANGES[feature]
            mid = (low + high) / 2
            
            if value == 0:
                deviation = 0
            else:
                deviation = (value - mid) / high
            
            contribution = deviation * importance * prediction
            contributions.append({
                'feature': FEATURE_NAMES[feature],
                'contribution': contribution
            })
    
    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    
    features = [c['feature'] for c in contributions]
    values = [c['contribution'] for c in contributions]
    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in values]
    
    bars = ax.barh(features, values, color=colors, height=0.6)
    ax.set_xlabel('Contribution to Prediction', fontsize=10)
    ax.set_title('Feature Contributions (Red = Risk, Green = Protective)', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    # Save to bytes
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer

def generate_pdf_report(data, result, risk_factors):
    """Generate PDF report"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 15, 'Diabetes Prediction Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 8, f'Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', ln=True, align='C')
    pdf.ln(10)
    
    # Divider
    pdf.set_draw_color(102, 126, 234)
    pdf.set_line_width(0.5)
    pdf.line(20, pdf.get_y(), 190, pdf.get_y())
    pdf.ln(10)
    
    # Prediction Result
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'PREDICTION RESULT', ln=True)
    pdf.ln(5)
    
    if result['prediction'] == 1:
        pdf.set_fill_color(231, 76, 60)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 12, f'  Status: DIABETIC  |  Confidence: {result["confidence"]:.1f}%', ln=True, fill=True)
    else:
        pdf.set_fill_color(39, 174, 96)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 12, f'  Status: NOT DIABETIC  |  Confidence: {result["confidence"]:.1f}%', ln=True, fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Patient Data
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'PATIENT DATA', ln=True)
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 10)
    data_items = [
        ('Pregnancies', f'{data[0]:.0f}'),
        ('Glucose', f'{data[1]:.1f} mg/dL'),
        ('Blood Pressure', f'{data[2]:.1f} mm Hg'),
        ('Skin Thickness', f'{data[3]:.1f} mm'),
        ('Insulin', f'{data[4]:.1f} mu U/ml'),
        ('BMI', f'{data[5]:.2f}'),
        ('Diabetes Pedigree', f'{data[6]:.3f}'),
        ('Age', f'{data[7]:.0f} years')
    ]
    
    for name, value in data_items:
        pdf.cell(60, 7, f'  {name}:', 0, 0)
        pdf.cell(0, 7, value, ln=True)
    
    pdf.ln(10)
    
    # Risk Analysis
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'RISK ANALYSIS', ln=True)
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 10)
    for factor in risk_factors[:5]:  # Top 5 risk factors
        color_map = {'High Risk': (231, 76, 60), 'Moderate Risk': (230, 126, 34), 
                     'Low': (243, 156, 18), 'Normal': (39, 174, 96), 'Concern': (243, 156, 18)}
        r, g, b = color_map.get(factor['level'], (0, 0, 0))
        
        pdf.set_text_color(r, g, b)
        pdf.cell(50, 7, f'  {factor["display_name"]}:', 0, 0)
        pdf.cell(30, 7, f'{factor["value"]:.1f}', 0, 0)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 7, f'[{factor["level"]}] - {factor["message"]}', ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    
    # Health Tips
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'HEALTH RECOMMENDATIONS', ln=True)
    pdf.ln(5)
    
    pdf.set_font('Helvetica', '', 10)
    tips = []
    
    if result['prediction'] == 1:
        tips = [
            '1. Consult a healthcare professional immediately',
            '2. Monitor blood sugar levels regularly',
            '3. Reduce intake of sugary and processed foods',
            '4. Exercise for at least 30 minutes daily',
            '5. Maintain a healthy weight (BMI 18.5-24.9)',
            '6. Get adequate sleep (7-8 hours)',
            '7. Manage stress through meditation or yoga'
        ]
    else:
        tips = [
            '1. Continue maintaining a healthy lifestyle',
            '2. Get regular health check-ups',
            '3. Stay physically active',
            '4. Eat a balanced diet rich in vegetables',
            '5. Maintain healthy weight',
            '6. Stay hydrated',
            '7. Monitor health annually'
        ]
    
    for tip in tips:
        pdf.cell(0, 7, f'  {tip}', ln=True)
    
    pdf.ln(10)
    
    # Disclaimer
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 7, 'Disclaimer: This report is generated by an AI model and should not replace', ln=True, align='C')
    pdf.cell(0, 7, 'professional medical advice. Please consult a doctor for proper diagnosis.', ln=True, align='C')
    
    return pdf

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
        
        # Analyze risk factors
        risk_factors = analyze_risk(data)
        
        # Generate contribution chart
        chart_buffer = generate_contribution_chart(data, prediction)
        chart_base64 = __import__('base64').b64encode(chart_buffer.getvalue()).decode()
        
        return render_template('index.html', result=result, values=dict(zip(FEATURES, data)),
                               risk_factors=risk_factors, chart_image=chart_base64)
    
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/download-report', methods=['POST'])
def download_report():
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
        
        # Analyze risk factors
        risk_factors = analyze_risk(data)
        
        # Generate PDF
        pdf = generate_pdf_report(data, result, risk_factors)
        
        # Save to bytes
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        
        return send_file(pdf_output, mimetype='application/pdf',
                        as_attachment=True, download_name='diabetes_report.pdf')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

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
