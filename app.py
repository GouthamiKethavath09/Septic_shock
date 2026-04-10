import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import shap
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

set_bg("img.jpg")

# ---------------- CONFIG ---------------- #
SEQ_LENGTH = 24
DEFAULT_AGE = 55

FEATURE_NAMES = [
    "bp","creatinine","heart_rate",
    "lactate","resp_rate","temperature","wbc","age"
]

# ---------------- LOAD ---------------- #
model = load_model("advanced_model.h5", compile=False)
scaler = pickle.load(open("Notebook/scaler.pkl", "rb"))

st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align:center;'>🧠 Septic Shock AI Dashboard</h1>", unsafe_allow_html=True)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📌 System Overview")
st.sidebar.info("""
🔬 AI-powered ICU system

✔ Predicts septic shock  
✔ Explains with SHAP  
✔ Shows attention patterns  
✔ Gives medical insights  

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
        text="Value"
    )

    st.plotly_chart(fig, use_container_width=True)

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

        # ---------------- METRICS ---------------- #
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Probability", f"{pred:.2f}")

        with col2:
            status = "HIGH RISK" if pred > 0.7 else "MODERATE" if pred > 0.4 else "LOW RISK"
            st.metric("Condition", status)

        with col3:
            st.metric("Confidence", f"{pred*100:.1f}%")

        # ================= SHAP ================= #
        st.markdown("## 🧠 SHAP Explainability")

        try:
            background = np.random.normal(np.mean(data_scaled), np.std(data_scaled)+1e-5, (50,24,8))
            explainer = shap.KernelExplainer(
                lambda x: model.predict(x.reshape(-1,24,8)),
                background.reshape(50,-1)
            )

            shap_values = explainer.shap_values(data_scaled.reshape(1,-1))

            shap_vals = shap_values[0].reshape(24,8)
            feature_impact = np.sum(np.abs(shap_vals), axis=0)

            shap_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Impact": feature_impact
            })

            st.bar_chart(shap_df.set_index("Feature"))

        except Exception as e:
            st.warning(f"SHAP error: {e}")

        # ---------------- SUMMARY ---------------- #
        summary = {
            "BP": df["bp"].iloc[-1],
            "Lactate": df["lactate"].iloc[-1],
            "Heart Rate": df["heart_rate"].iloc[-1],
            "WBC": df["wbc"].iloc[-1]
        }

        st.table(pd.DataFrame(summary.items(), columns=["Parameter","Value"]))

        show_comparison(df)

        # ---------------- GAUGE ---------------- #
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=pred))
        st.plotly_chart(gauge)

        # ---------------- TRENDS ---------------- #
        df_plot = pd.DataFrame(data, columns=FEATURE_NAMES)
        st.line_chart(df_plot[["bp","heart_rate","lactate","wbc"]])

        # ---------------- MEDICAL INSIGHTS ---------------- #
        st.markdown("## 🧠 Medical Insights")

        insights = []

        if df["lactate"].iloc[-1] > 2.5:
            insights.append("High lactate → tissue hypoxia")

        if df["bp"].iloc[-1] < 90:
            insights.append("Low BP → shock condition")

        if df["heart_rate"].iloc[-1] > 110:
            insights.append("High HR → stress response")

        if df["wbc"].iloc[-1] > 12:
            insights.append("High WBC → infection")

        for i in insights:
            st.write(f"👉 {i}")

        # ---------------- DOWNLOAD REPORT ---------------- #
        report = f"""
Septic Shock Report

Risk Score: {pred:.2f}
Status: {status}

Insights:
{insights}
"""

        st.download_button("📄 Download Report", report, "report.txt")

        # ---------------- FINAL ---------------- #
        if pred > 0.7:
            st.error("⚠️ ICU required")
        elif pred > 0.4:
            st.warning("Monitor closely")
        else:
            st.success("Stable")
