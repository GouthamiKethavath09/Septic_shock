# 🧠 Septic Shock Prediction System  
### AI-Powered ICU Decision Support Dashboard

---
## live link: https://septicshockpedictor-hj9vzn62y4scakhtwvbydy.streamlit.app/
## 📌 Overview
Septic shock is a life-threatening medical condition caused by severe infection, leading to organ failure and dangerously low blood pressure. Early detection is critical to improving survival rates.

This project presents an AI-powered clinical decision support system that predicts the risk of septic shock using time-series ICU patient data. The system integrates Deep Learning (LSTM) with an interactive Streamlit dashboard to provide real-time risk prediction, clinical insights, visual analytics, and actionable medical recommendations.

---

## 🎯 Objective
The main objective of this project is to assist healthcare professionals by:
- Detecting early signs of septic shock  
- Monitoring patient vitals continuously  
- Supporting faster and more accurate clinical decisions  
- Improving patient outcomes in ICU environments  

---

## 🧠 System Workflow
1. Data Input  
   - Time-series ICU patient data (24 time steps)  
   - Multiple physiological parameters  

2. Data Preprocessing  
   - Normalization using MinMaxScaler  
   - Feature alignment  
   - Sequence creation for LSTM  

3. Model Prediction  
   - LSTM model captures temporal patterns  
   - Outputs probability of septic shock  

4. Visualization & Insights  
   - Risk meter (gauge chart)  
   - Vital trends  
   - Patient vs normal comparison  
   - AI-based medical interpretation  

---

## 🧪 Input Data Format
The model expects a CSV file with:
- 24 rows (time steps)  
- 7 features  

### Features:
- Blood Pressure (bp)  
- Creatinine  
- Heart Rate  
- Lactate  
- Respiration Rate  
- Temperature  
- White Blood Cell Count (WBC)  

---

## 🧠 Model Architecture
- Model Type: LSTM (Long Short-Term Memory)  
- Input Shape: (24, 8)  
- Layers:
  - LSTM (64 units)  
  - Dropout  
  - LSTM (32 units)  
  - Dense (ReLU)  
  - Output Layer (Sigmoid)  

### Why LSTM?
LSTM is ideal for time-series healthcare data because it captures temporal dependencies and patterns across patient vitals over time.

---

## 📊 Key Features

### 🔍 Risk Prediction
- Predicts probability of septic shock  
- Classifies into:
  - Low Risk  
  - Moderate Risk  
  - High Risk  

---

### 📊 Patient vs Normal Comparison
- Compares patient vitals with standard medical values  
- Highlights abnormal conditions clearly  
- Improves interpretability  

---

### 📈 Vital Trends Visualization
- Displays time-series trends of:
  - Blood Pressure  
  - Heart Rate  
  - Lactate  
  - WBC  

---

### 🩺 Risk Meter (Gauge Chart)
- Visual representation of risk level  
- Color-coded:
  - Green → Stable  
  - Orange → Warning  
  - Red → Critical  

---

### 🧠 AI-Based Medical Insights
Automatically identifies:
- Tissue hypoxia (high lactate)  
- Hypotension (low BP)  
- Infection (high WBC)  
- Stress response (high heart rate)  

---

### 🛡️ Clinical Recommendations
Provides actionable suggestions:
- Start IV fluids  
- Administer vasopressors  
- Monitor cardiovascular status  
- Begin antibiotics  

---

### 📋 Clinical Summary
- Displays latest patient values  
- Helps quick medical assessment  

---

### 📄 Report Generation
- Downloadable patient report  
- Includes risk score, insights, and diagnosis  

---

### 🎨 User Interface
- Glassmorphism design  
- Interactive charts using Plotly  
- Clean and responsive dashboard  
- Sidebar system overview  

---

## ⚙️ Tech Stack

### Machine Learning
- TensorFlow / Keras  
- LSTM (Deep Learning)

### Data Processing
- NumPy  
- Pandas  
- Scikit-learn  

### Visualization
- Plotly  
- Streamlit  

### Frontend
- Streamlit  

---


## 📊 Model Performance
- Accuracy: ~80%  
- AUROC: ~0.87  
- Effective for time-series ICU prediction  

---

## 💡 Future Enhancements
- SHAP Explainability  
- Real-time ICU monitoring  
- Multi-patient dashboard  
- Alert notification system  
- Integration with hospital systems  

---

