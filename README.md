# Healthcare Predictive Analytics - Diabetes Prediction

A machine learning system that predicts diabetes risk using patient medical data. Built with Python, Scikit-learn, XGBoost, and deployed with Flask.

**Live Link:** https://healthcare-predictive-analytics.onrender.com

## Features

- **4 ML Models Compared**: Logistic Regression, Random Forest, SVM, XGBoost
- **Data Visualizations**: Correlation heatmap, feature importance, box plots, scatter plots
- **SHAP/LIME Explainability**: Understand why the model makes predictions
- **Flask Web App**: Interactive UI for making predictions
- **REST API**: Programmatic access to predictions

## Tech Stack

- Python 3.9
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- SHAP, LIME
- Flask

## Run Locally

```bash
pip install -r requirements.txt
python train.py
python app.py
```

Open http://localhost:5000

## API Usage

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```
