# 🧠 Septic Shock Prediction System using Deep Learning & Explainable AI

---

## 📌 Project Overview

Septic shock is a life-threatening medical condition caused by severe infection leading to dangerously low blood pressure and organ failure. Early prediction and timely intervention are critical to saving lives.
### Live link: https://septicshock-svnrq6nilpywkisvappppdzq.streamlit.app/
This project presents a **complete AI-powered healthcare system** that predicts septic shock risk using **time-series patient data**. It combines **advanced deep learning models (CNN + BiLSTM + Attention)** with **Explainable AI (SHAP)** to provide both accurate predictions and meaningful clinical insights.

The system is deployed using **Streamlit**, offering a real-time, interactive dashboard for doctors, researchers, and healthcare professionals.

---

## 🎯 Objectives

- Develop a deep learning model to predict septic shock risk  
- Utilize time-series patient data for better accuracy  
- Integrate Explainable AI (SHAP) for transparency  
- Provide visual insights for clinical decision-making  
- Build a deployable end-to-end healthcare AI system  

---

## 🧠 Model Architecture (Advanced Deep Learning)

The model is designed using a hybrid deep learning architecture:

### 🔹 1. CNN (Convolutional Neural Network)
- Extracts local temporal features from patient vitals  
- Captures short-term fluctuations and patterns  

### 🔹 2. BiLSTM (Bidirectional Long Short-Term Memory)
- Learns long-term dependencies in sequential data  
- Processes data forward and backward  
- Improves understanding of patient trends  

### 🔹 3. Attention Mechanism
- Assigns importance weights to different time steps  
- Focuses on critical medical conditions  
- Enhances interpretability  

### 🔹 4. Dense Layers
- Fully connected layers for classification  
- Final output uses sigmoid activation  

---

## 🧬 Explainable AI (SHAP Integration)

To make predictions interpretable, SHAP (SHapley Additive exPlanations) is used.

### 🔥 SHAP Features Implemented:

- Feature Importance (global impact)
- SHAP Waterfall (individual feature contribution)
- Time-wise SHAP (impact across 24 time steps)
- Positive vs Negative influence
- Feature contribution distribution (pie chart)
- AI-based medical explanations

### 🧠 Why SHAP?

- Helps doctors understand *why* a prediction was made  
- Increases trust in AI systems  
- Bridges gap between AI and healthcare  

---

## 📊 Features Used

| Feature        | Description                        |
|----------------|----------------------------------|
| BP             | Blood Pressure                   |
| Creatinine     | Kidney function indicator        |
| Heart Rate     | Cardiovascular activity          |
| Lactate        | Tissue oxygen level              |
| Resp Rate      | Breathing rate                   |
| Temperature    | Body temperature                 |
| WBC            | Infection indicator              |
| Age            | Patient age                      |

---

## 📁 Input Data Format

- CSV file with **24 rows × 7 columns**
- Each row = one time step
- Represents patient vitals over time
- Age is added automatically during preprocessing

---

## ⚙️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas**
- **SHAP (Explainable AI)**
- **Plotly (Visualization)**
- **Streamlit (Deployment UI)**

---

