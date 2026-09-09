import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib
import os

print("=" * 60)
print("HEALTHCARE PREDICTIVE ANALYTICS - DIABETES PREDICTION")
print("=" * 60)

# Create charts directory
os.makedirs('static/charts', exist_ok=True)

# ============================================
# STEP 1: LOAD DATA
# ============================================
print("\n[1/8] Loading dataset...")
df = pd.read_csv('data/diabetes.csv')
print(f"Dataset loaded: {df.shape[0]} patients, {df.shape[1]} features")
print(f"\nFeatures: {list(df.columns)}")

# ============================================
# STEP 2: DATA CLEANING
# ============================================
print("\n[2/8] Cleaning data...")

# Replace 0 values with NaN for columns where 0 is not valid
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)

# Fill missing values with median
for col in zero_columns:
    df[col] = df[col].fillna(df[col].median())

print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# ============================================
# STEP 3: DATA VISUALIZATION
# ============================================
print("\n[3/8] Creating visualizations...")

# 1. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('static/charts/correlation_heatmap.png', dpi=100)
plt.close()
print("  - Correlation heatmap saved")

# 2. Disease Distribution
plt.figure(figsize=(6, 4))
df['Outcome'].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Diabetes Distribution', fontsize=14)
plt.xlabel('Outcome (0=No Diabetes, 1=Diabetes)')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('static/charts/disease_distribution.png', dpi=100)
plt.close()
print("  - Disease distribution saved")

# 3. Age Distribution
plt.figure(figsize=(8, 5))
plt.hist(df['Age'], bins=30, color='#3498db', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('static/charts/age_distribution.png', dpi=100)
plt.close()
print("  - Age distribution saved")

# 4. Age vs Blood Pressure (Scatter)
plt.figure(figsize=(8, 5))
colors = ['#2ecc71' if x == 0 else '#e74c3c' for x in df['Outcome']]
plt.scatter(df['Age'], df['BloodPressure'], c=colors, alpha=0.6)
plt.title('Age vs Blood Pressure', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Blood Pressure')
plt.legend(['No Diabetes', 'Diabetes'], loc='upper right')
plt.tight_layout()
plt.savefig('static/charts/age_vs_bp.png', dpi=100)
plt.close()
print("  - Age vs BP scatter saved")

# 5. Feature Spread (Box Plot)
plt.figure(figsize=(12, 6))
df_melted = df.melt(id_vars='Outcome', value_vars=['Glucose', 'BMI', 'Age', 'Insulin'])
sns.boxplot(data=df_melted, x='variable', y='value', hue='Outcome', palette=['#2ecc71', '#e74c3c'])
plt.title('Feature Distribution by Diabetes Status', fontsize=14)
plt.tight_layout()
plt.savefig('static/charts/feature_boxplot.png', dpi=100)
plt.close()
print("  - Feature boxplot saved")

# ============================================
# STEP 4: PREPARE DATA FOR ML
# ============================================
print("\n[4/8] Preparing data for ML...")

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# ============================================
# STEP 5: TRAIN MODELS
# ============================================
print("\n[5/8] Training 4 ML models...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = {}
for name, model in models.items():
    print(f"  Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results[name] = {'accuracy': acc, 'f1': f1, 'model': model, 'predictions': y_pred}
    print(f"    Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# ============================================
# STEP 6: MODEL COMPARISON
# ============================================
print("\n[6/8] Comparing models...")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")

# Model Comparison Bar Chart
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', color='#3498db')
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='#2ecc71')
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('Model Comparison', fontsize=14)
plt.xticks(x, model_names, rotation=15)
plt.legend()
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig('static/charts/model_comparison.png', dpi=100)
plt.close()
print("  - Model comparison chart saved")

# Confusion Matrix for best model
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('static/charts/confusion_matrix.png', dpi=100)
plt.close()
print("  - Confusion matrix saved")

# ============================================
# STEP 7: FEATURE IMPORTANCE (Random Forest)
# ============================================
print("\n[7/8] Analyzing feature importance...")

rf_model = results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

plt.figure(figsize=(8, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='#9b59b6')
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('static/charts/feature_importance.png', dpi=100)
plt.close()
print("  - Feature importance chart saved")

# ============================================
# STEP 8: SAVE MODEL AND SCALER
# ============================================
print("\n[8/8] Saving best model and scaler...")

joblib.dump(results[best_model_name]['model'], 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print(f"  Model saved: best_model.pkl ({best_model_name})")
print(f"  Scaler saved: scaler.pkl")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nBest Model: {best_model_name}")
print(f"Accuracy: {best_accuracy:.4f}")
print(f"F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"\nAll models compared:")
for name in results:
    print(f"  {name}: Accuracy={results[name]['accuracy']:.4f}, F1={results[name]['f1']:.4f}")
print(f"\nCharts saved in: static/charts/")
print(f"Model saved as: best_model.pkl")
print("=" * 60)
