import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# ---------------- BACKGROUND + GLASS ---------------- #
def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-attachment: fixed;
    }}

    .glass {{
        background: rgba(255,255,255,0.15);
        padding: 20px;
        border-radius: 15px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.3);
        color: white;
    }}

    h1,h2,h3,h4,p,li {{
        color: white !important;
    }}

    section[data-testid="stSidebar"] * {{
        color: black !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("C:\\Users\\hp\\Desktop\\Septic_shock_prediction\\img.jpg")

# ---------------- CONFIG ---------------- #
SEQ_LENGTH = 24
DEFAULT_AGE = 55

FEATURE_NAMES = [
    "bp","creatinine","heart_rate",
    "lactate","resp_rate","temperature","wbc","age"
]

# ---------------- LOAD ---------------- #
model = load_model("C:\\Users\\hp\\Desktop\\Septic_shock_prediction\\advanced_model.h5", compile=False)
scaler = pickle.load(open("C:\\Users\\hp\\Desktop\\Septic_shock_prediction\\Notebook\\scaler.pkl", "rb"))

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align:center;'>🧠 Septic Shock AI Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📌 System Overview")
st.sidebar.info("""
🔬 AI-powered ICU system

✔ Predicts septic shock risk  
✔ Analyzes patient vitals  
✔ Provides medical insights  
✔ Suggests precautions  

📊 Features Used:
- Blood Pressure (BP)
- Creatinine
- Heart Rate
- Lactate
- Respiration Rate
- Temperature
- WBC Count
- Age
""")

# ---------------- COMPARISON FUNCTION ---------------- #
def show_comparison(df):
    st.markdown("## 📊 Patient vs Normal Comparison")

    normal = {
        "BP": 120,
        "Heart Rate": 75,
        "Lactate": 1.0,
        "WBC": 7
    }

    current = {
        "BP": df["bp"].iloc[-1],
        "Heart Rate": df["heart_rate"].iloc[-1],
        "Lactate": df["lactate"].iloc[-1],
        "WBC": df["wbc"].iloc[-1]
    }

    comp_df = pd.DataFrame({
        "Parameter": list(normal.keys()),
        "Patient": list(current.values()),
        "Normal": list(normal.values())
    })

    comp_melt = comp_df.melt(id_vars="Parameter", var_name="Type", value_name="Value")

    fig = px.bar(
        comp_melt,
        x="Parameter",
        y="Value",
        color="Type",
        barmode="group",
        text="Value",
        title="Patient vs Normal Values"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🧠 Interpretation")

    if current["BP"] < 90:
        st.write("• 🔴 Blood Pressure is low → possible shock")

    if current["Lactate"] > 2.5:
        st.write("• 🔴 Lactate is high → tissue hypoxia")

    if current["Heart Rate"] > 110:
        st.write("• 🟠 Heart Rate is elevated → stress response")

    if current["WBC"] > 12:
        st.write("• 🔴 WBC is high → infection likely")

# ---------------- UPLOAD ---------------- #
st.markdown("## 📂 Upload Patient Data")
file = st.file_uploader("Upload CSV (24×7)", type=["csv"])

data_array = None

if file:
    df = pd.read_csv(file)

    if df.shape != (24,7):
        st.error("❌ CSV must be 24 rows × 7 columns")
    else:
        st.success("✅ Data Loaded")
        st.dataframe(df)
        data_array = df.values

# ---------------- PREDICT ---------------- #
if st.button("🚀 Analyze Patient"):

    if data_array is None:
        st.warning("Upload data first")

    else:
        age_col = np.full((SEQ_LENGTH,1), DEFAULT_AGE)
        data = np.hstack([data_array, age_col])

        data_scaled = scaler.transform(data)
        data_scaled = data_scaled.reshape(1,24,8)

        pred = model.predict(data_scaled)[0][0]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("<div class='glass'><h3>Risk Score</h3></div>", unsafe_allow_html=True)
            st.metric("Probability", f"{pred:.2f}")

        with col2:
            if pred > 0.7:
                status = "HIGH RISK"
            elif pred > 0.4:
                status = "MODERATE"
            else:
                status = "LOW RISK"
            st.markdown("<div class='glass'><h3>Status</h3></div>", unsafe_allow_html=True)
            st.metric("Condition", status)

        with col3:
            st.markdown("<div class='glass'><h3>Confidence</h3></div>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{pred*100:.1f}%")

        # ================= SHAP ================= #
        import shap
        np.int = int

        st.subheader("🧠 AI Explainability (SHAP)")

        try:
            background = data_scaled.reshape(1, -1)
            sample = data_scaled.reshape(1, -1)

            def predict_fn(x):
                return model.predict(x.reshape(-1, 24, 8))

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(sample)

            feature_names = df.columns.tolist()

            shap_vals = shap_values[0][:len(feature_names)]

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Impact": np.abs(shap_vals)
            }).sort_values(by="Impact", ascending=False)

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                x=shap_df["Impact"],
                y=shap_df["Feature"],
                orientation='h'
            ))

            st.plotly_chart(fig_shap, use_container_width=True)

        except:
            st.warning("SHAP error")

        # ================= WATERFALL ================= #
        st.subheader("🔥 SHAP Waterfall")

        try:
            fig_waterfall = go.Figure(go.Waterfall(
                y=feature_names,
                x=shap_vals,
                orientation="h"
            ))
            st.plotly_chart(fig_waterfall, use_container_width=True)
        except:
            st.warning("Waterfall error")

        # ================= RISK TRACK ================= #
        st.subheader("📈 Risk Tracking")

        try:
            risk_values = []

            for i in range(5, 25):
                temp_data = df.iloc[:i].values

                if temp_data.shape[0] < 24:
                    pad = np.zeros((24 - temp_data.shape[0], 8))
                    temp_data = np.vstack([pad, temp_data])

                temp_scaled = scaler.transform(temp_data).reshape(1, 24, 8)
                temp_pred = model.predict(temp_scaled)[0][0]

                risk_values.append(temp_pred)

            fig_risk = go.Figure()
            fig_risk.add_trace(go.Scatter(y=risk_values, mode='lines+markers'))

            st.plotly_chart(fig_risk, use_container_width=True)

        except:
            st.warning("Risk tracking error")

        # ================= SUMMARY ================= #
        st.markdown("<div class='glass'><h3>📋 Clinical Summary</h3></div>", unsafe_allow_html=True)

        summary = {
            "BP": df["bp"].iloc[-1],
            "Lactate": df["lactate"].iloc[-1],
            "Heart Rate": df["heart_rate"].iloc[-1],
            "WBC": df["wbc"].iloc[-1]
        }

        st.table(pd.DataFrame(summary.items(), columns=["Parameter","Value"]))

        show_comparison(df)

        # ================= GAUGE ================= #
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            gauge={'axis': {'range': [0,1]}}
        ))

        st.plotly_chart(gauge, use_container_width=True)

        # ================= FINAL ================= #
        if pred > 0.7:
            st.error("⚠️ Immediate ICU intervention required")
        elif pred > 0.4:
            st.warning("🟠 Monitor closely")
        else:
            st.success("✅ Stable condition")