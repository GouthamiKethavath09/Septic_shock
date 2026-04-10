import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64
import shap
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

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

        # ---------------- TOP METRICS ---------------- #
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
        st.markdown("## 🧠 AI Explainability (SHAP)")

        try:
            background = np.random.normal(
                loc=np.mean(data_scaled),
                scale=np.std(data_scaled) + 1e-5,
                size=(50,24,8)
            )

            background = background.reshape(50, -1)
            sample = data_scaled.reshape(1, -1)

            def predict_fn(x):
                return model.predict(x.reshape(-1,24,8))

            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(sample)

            shap_vals = shap_values[0].flatten()
            shap_vals = shap_vals.reshape(24,8)
            shap_vals = np.sum(np.abs(shap_vals), axis=0)

            shap_df = pd.DataFrame({
                "Feature": FEATURE_NAMES,
                "Impact": shap_vals
            }).sort_values(by="Impact", ascending=True)

            fig_shap = px.bar(
                shap_df,
                x="Impact",
                y="Feature",
                orientation='h',
                color="Impact",
                color_continuous_scale="Reds",
                title="🔥 Feature Importance"
            )

            st.plotly_chart(fig_shap, use_container_width=True)

            # Waterfall
            st.subheader("🔥 SHAP Waterfall")

            fig_w = go.Figure(go.Waterfall(
                y=FEATURE_NAMES,
                x=shap_vals,
                orientation="h"
            ))

            st.plotly_chart(fig_w, use_container_width=True)

            # Top drivers
            st.subheader("📌 Key Drivers")

            for i in range(3):
                st.write(f"👉 {shap_df.iloc[-(i+1)]['Feature']} strongly influenced prediction")

            # ================= EXTRA SHAP ================= #

            st.subheader("⏳ Time-wise Risk Contribution")

            shap_time = shap_values[0].reshape(24, 8)
            time_impact = np.sum(np.abs(shap_time), axis=1)

            fig_time = px.line(
                y=time_impact,
                title="Risk Contribution Across 24 Time Steps",
                markers=True
            )
            st.plotly_chart(fig_time, use_container_width=True)

            st.subheader("⚖️ Positive vs Negative Influence")

            positive = np.sum(shap_time[shap_time > 0])
            negative = np.sum(shap_time[shap_time < 0])

            fig_pn = go.Figure(data=[
                go.Bar(name="Positive Impact", x=["Impact"], y=[positive]),
                go.Bar(name="Negative Impact", x=["Impact"], y=[abs(negative)])
            ])
            st.plotly_chart(fig_pn, use_container_width=True)

            st.subheader("🧩 Contribution Distribution")

            fig_pie = px.pie(
                shap_df,
                names="Feature",
                values="Impact"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP error: {e}")

        # ---------------- SUMMARY ---------------- #
        st.markdown("<div class='glass'><h3>📋 Clinical Summary</h3></div>", unsafe_allow_html=True)

        summary = {
            "BP": df["bp"].iloc[-1],
            "Lactate": df["lactate"].iloc[-1],
            "Heart Rate": df["heart_rate"].iloc[-1],
            "WBC": df["wbc"].iloc[-1]
        }

        st.table(pd.DataFrame(summary.items(), columns=["Parameter","Value"]))

        show_comparison(df)

        # ---------------- FINAL ---------------- #
        if pred > 0.7:
            st.error("⚠️ Immediate ICU intervention required")
        elif pred > 0.4:
            st.warning("🟠 Monitor closely")
        else:
            st.success("✅ Stable condition")
        st.markdown("<div class='glass'><h3>🧠 Medical Insights</h3></div>", unsafe_allow_html=True)

        insights = []
        precautions = []

        if df["lactate"].iloc[-1] > 2.5:
            insights.append("High lactate → tissue hypoxia")

        if df["bp"].iloc[-1] < 90:
            insights.append("Low BP → shock condition")

        if df["heart_rate"].iloc[-1] > 110:
            insights.append("High HR → stress response")

        if df["wbc"].iloc[-1] > 12:
            insights.append("High WBC → infection")

        precautions = [
            "Start IV fluids",
            "Administer vasopressors",
            "Monitor heart",
            "Start antibiotics"
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Conditions")
            for i in insights:
                st.markdown(f"- {i}")

        with col2:
            st.markdown("### 🛡️ Precautions")
            for p in precautions:
                st.markdown(f"- {p}")
def create_pdf(pred, status, insights, precautions):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Septic Shock AI Report", styles['Title']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Risk Score: {pred:.2f}", styles['Normal']))
    story.append(Paragraph(f"Status: {status}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Medical Insights:", styles['Heading2']))
    for i in insights:
        story.append(Paragraph(f"- {i}", styles['Normal']))

    story.append(Spacer(1, 12))

    story.append(Paragraph("Precautions:", styles['Heading2']))
    for p in precautions:
        story.append(Paragraph(f"- {p}", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer
pdf_file = create_pdf(pred, status, insights, precautions)

st.download_button(
    label="📥 Download PDF Report",
    data=pdf_file,
    file_name="Septic_Shock_Report.pdf",
    mime="application/pdf"
)



