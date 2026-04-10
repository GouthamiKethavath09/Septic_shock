import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import shap
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px

# ================= CONFIG FIRST ================= #
st.set_page_config(layout="wide")
np.int = int

# ---------------- BACKGROUND ---------------- #
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
    }}
    h1,h2,h3,h4,p,li {{ color: white !important; }}
    section[data-testid="stSidebar"] * {{ color: black !important; }}
    </style>
    """, unsafe_allow_html=True)

set_bg("img.jpg")

# ---------------- LOAD ---------------- #
model = load_model("advanced_model.h5", compile=False)
scaler = pickle.load(open("Notebook/scaler.pkl", "rb"))

SEQ_LENGTH = 24
DEFAULT_AGE = 55

FEATURE_NAMES = [
    "BP","Creatinine","Heart Rate",
    "Lactate","Resp Rate","Temperature","WBC","Age"
]

# ---------------- UI ---------------- #
st.title("🧠 Septic Shock AI Dashboard")

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("📌 System Overview")
st.sidebar.info("""
AI-powered ICU system

✔ Predicts septic shock  
✔ Explains with SHAP  
✔ Shows attention patterns  
✔ Gives medical insights  
""")

# ---------------- UPLOAD ---------------- #
file = st.file_uploader("Upload CSV (24×7)", type=["csv"])

if file:
    df = pd.read_csv(file)

    if df.shape != (24,7):
        st.error("❌ CSV must be 24x7")
    else:
        st.success("✅ Data Loaded")
        st.dataframe(df)

        if st.button("🚀 Analyze Patient"):

            # ---------------- PREPROCESS ---------------- #
            age_col = np.full((SEQ_LENGTH,1), DEFAULT_AGE)
            data = np.hstack([df.values, age_col])
            data_scaled = scaler.transform(data).reshape(1,24,8)

            pred = model.predict(data_scaled)[0][0]

            # ---------------- METRICS ---------------- #
            c1,c2,c3 = st.columns(3)
            c1.metric("Risk Score", f"{pred:.2f}")

            if pred > 0.7:
                status = "HIGH RISK"
            elif pred > 0.4:
                status = "MODERATE"
            else:
                status = "LOW RISK"

            c2.metric("Status", status)
            c3.metric("Confidence", f"{pred*100:.1f}%")

            # ================= SHAP ================= #
            st.markdown("## 🧠 SHAP Explainability")

            try:
                background = np.random.normal(
                    np.mean(data_scaled),
                    np.std(data_scaled)+1e-5,
                    size=(50,24,8)
                )

                explainer = shap.KernelExplainer(
                    lambda x: model.predict(x.reshape(-1,24,8)),
                    background.reshape(50,-1)
                )

                shap_values = explainer.shap_values(data_scaled.reshape(1,-1))

                shap_vals = shap_values[0].reshape(24,8)
                shap_vals = np.sum(np.abs(shap_vals), axis=0)

                shap_df = pd.DataFrame({
                    "Feature": FEATURE_NAMES,
                    "Impact": shap_vals
                }).sort_values(by="Impact")

                fig = px.bar(
                    shap_df, x="Impact", y="Feature",
                    orientation='h',
                    color="Impact",
                    color_continuous_scale="Reds"
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"SHAP issue: {e}")

            # ================= ATTENTION HEATMAP ================= #
            st.markdown("## 🔥 Attention Heatmap")

            try:
                attention = np.mean(data_scaled[0], axis=1)

                fig_att = px.imshow(
                    attention.reshape(24,1),
                    labels=dict(x="Attention", y="Time Step"),
                    color_continuous_scale="Viridis"
                )

                st.plotly_chart(fig_att, use_container_width=True)

            except:
                st.warning("Attention not available")

            # ================= COMPARISON ================= #
            st.markdown("## 📊 Patient vs Normal")

            normal = {"bp":120,"heart_rate":75,"lactate":1,"wbc":7}
            current = {
                "bp":df["bp"].iloc[-1],
                "heart_rate":df["heart_rate"].iloc[-1],
                "lactate":df["lactate"].iloc[-1],
                "wbc":df["wbc"].iloc[-1]
            }

            comp_df = pd.DataFrame({
                "Parameter": list(normal.keys()),
                "Patient": list(current.values()),
                "Normal": list(normal.values())
            })

            st.bar_chart(comp_df.set_index("Parameter"))

            # ================= RISK METER ================= #
            st.markdown("## 🩺 Risk Meter")

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                gauge={'axis': {'range': [0,1]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)

            # ================= TRENDS ================= #
            st.markdown("## 📈 Vital Trends")

            df_plot = pd.DataFrame(data, columns=FEATURE_NAMES)
            st.line_chart(df_plot[["bp","heart_rate","lactate","wbc"]])

            # ================= INSIGHTS ================= #
            st.markdown("## 🧠 Medical Insights")

            if df["lactate"].iloc[-1] > 2.5:
                st.error("High lactate → tissue hypoxia")
            if df["bp"].iloc[-1] < 90:
                st.error("Low BP → shock risk")

            # ================= FINAL ================= #
            st.markdown("## 📌 Final Diagnosis")

            if pred > 0.7:
                st.error("⚠️ ICU Required")
            elif pred > 0.4:
                st.warning("Monitor closely")
            else:
                st.success("Stable")

            # ================= DOWNLOAD ================= #
            report = f"Risk: {pred}\nStatus: {status}"
            st.download_button("Download Report", report)
