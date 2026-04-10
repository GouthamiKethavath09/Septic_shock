import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
import shap

# ================= MUST BE FIRST ================= #
st.set_page_config(layout="wide")

# FIX SHAP + NUMPY
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
    h1,h2,h3,h4,p,li {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: black !important;
    }}
    </style>
    """, unsafe_allow_html=True)

set_bg("img.jpg")

# ---------------- LOAD ---------------- #
model = load_model("advanced_model.h5", compile=False)
scaler = pickle.load(open("Notebook/scaler.pkl", "rb"))

SEQ_LENGTH = 24
DEFAULT_AGE = 55

# ---------------- UI ---------------- #
st.title("🧠 Septic Shock AI Dashboard")

file = st.file_uploader("Upload CSV (24x7)", type=["csv"])

if file:
    df = pd.read_csv(file)

    if df.shape != (24,7):
        st.error("❌ CSV must be 24x7")
    else:
        st.success("✅ Data Loaded")
        st.dataframe(df)

        if st.button("🚀 Analyze Patient"):

            # ---------- PREPROCESS ---------- #
            age_col = np.full((SEQ_LENGTH,1), DEFAULT_AGE)
            data = np.hstack([df.values, age_col])

            data_scaled = scaler.transform(data)
            data_scaled = data_scaled.reshape(1,24,8)

            pred = model.predict(data_scaled)[0][0]

            st.metric("Risk Score", f"{pred:.2f}")

            # ================= SHAP ================= #
            st.subheader("🧠 SHAP Explainability")

            try:
                # USE RANDOM BACKGROUND (IMPORTANT)
                background = np.tile(data_scaled, (20,1,1))
                background = background.reshape(20, -1)

                sample = data_scaled.reshape(1, -1)

                def predict_fn(x):
                    return model.predict(x.reshape(-1,24,8))

                explainer = shap.KernelExplainer(predict_fn, background)

                shap_values = explainer.shap_values(sample)

                feature_names = df.columns.tolist()

                shap_vals = shap_values[0][:len(feature_names)]

                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Impact": np.abs(shap_vals)
                }).sort_values(by="Impact", ascending=False)

                # -------- BAR CHART -------- #
                fig = px.bar(
                    shap_df,
                    x="Impact",
                    y="Feature",
                    orientation='h',
                    title="Feature Importance"
                )
                st.plotly_chart(fig, use_container_width=True)

                # -------- WATERFALL -------- #
                st.subheader("🔥 SHAP Waterfall")

                fig2 = go.Figure(go.Waterfall(
                    y=feature_names,
                    x=shap_vals,
                    orientation="h"
                ))
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error("SHAP error: " + str(e))

            # ================= RISK TRACK ================= #
            st.subheader("📈 Risk Trend")

            try:
                risk_values = []

                for i in range(5,25):
                    temp = df.iloc[:i].values

                    if temp.shape[0] < 24:
                        pad = np.zeros((24-temp.shape[0],7))
                        temp = np.vstack([pad, temp])

                    temp = np.hstack([temp, np.full((24,1), DEFAULT_AGE)])

                    temp_scaled = scaler.transform(temp).reshape(1,24,8)

                    risk_values.append(model.predict(temp_scaled)[0][0])

                st.line_chart(risk_values)

            except:
                st.warning("Risk tracking error")
