import streamlit as st
import requests
import matplotlib.pyplot as plt
from reportlab.platypus import Image
import os

# PDF imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# --------------------------------------------------
# CLINICAL INTERPRETATION
# --------------------------------------------------
def clinical_interpretation(disease, patient_data, risk_score, decision):
    if decision == "Low Risk":
        return (
            "Based on the provided clinical parameters, the patient currently "
            "falls into a low-risk category. The values are within or close to "
            "normal reference ranges. Routine monitoring is advised."
        )
    elif decision == "Moderate Risk":
        return (
            "The patient shows moderate risk. Some parameters approach "
            "clinical thresholds. Lifestyle modification and follow-up "
            "screening are recommended."
        )
    else:
        return (
            "The patient is at high risk. Multiple parameters exceed "
            "clinical thresholds. Further diagnostic testing and "
            "clinical consultation are strongly advised."
        )

# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------
def generate_pdf_report(
    filename,
    disease,
    patient_data,
    risk_score,
    decision,
    chart_path
):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=A4)
    elements = []

    elements.append(Paragraph("<b>Clinical AI Risk Assessment Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"<b>Condition:</b> {disease.capitalize()}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    table_data = [["Parameter", "Value"]]
    for k, v in patient_data.items():
        table_data.append([k, str(v)])

    table = Table(table_data, colWidths=[220, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    elements.append(Paragraph(f"<b>Risk Score:</b> {risk_score:.2f}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Clinical Risk Level:</b> {decision}", styles["Normal"]))
    elements.append(Spacer(1, 16))

    if chart_path and os.path.exists(chart_path):
        elements.append(Paragraph("<b>Clinical Parameter Overview</b>", styles["Heading2"]))
        elements.append(Spacer(1, 8))
        elements.append(Image(chart_path, width=400, height=250))
        elements.append(Spacer(1, 16))

    elements.append(Paragraph("<b>Clinical Interpretation</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(
        clinical_interpretation(disease, patient_data, risk_score, decision),
        styles["Normal"]
    ))

    elements.append(Spacer(1, 16))
    elements.append(Paragraph(
        "<i>Disclaimer: Educational and decision-support use only.</i>",
        styles["Italic"]
    ))

    doc.build(elements)

# --------------------------------------------------
# STREAMLIT PAGE
# --------------------------------------------------
st.set_page_config(page_title="Clinical AI Risk Assessment", layout="centered")

st.markdown(
    """
    <div style="background:#0f4c81;padding:18px;border-radius:14px;">
        <h2 style="color:white;">Clinical AI Risk Assessment System</h2>
        <p style="color:#e5f0ff;">Agentic AI + ML for EHR-based risk prediction</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "disease" not in st.session_state:
    st.session_state.disease = "diabetes"
if "result" not in st.session_state:
    st.session_state.result = None
if "risk_score" not in st.session_state:
    st.session_state.risk_score = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None

# --------------------------------------------------
# DISEASE SELECTION
# --------------------------------------------------
st.markdown("### Select Condition")
c1, c2 = st.columns(2)

with c1:
    if st.button("🩸 Diabetes"):
        st.session_state.disease = "diabetes"

with c2:
    if st.button("❤️ Heart Disease"):
        st.session_state.disease = "heart"

disease = st.session_state.disease

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.markdown("### Patient Information")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", 1, 120, 50)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

with c2:
    if disease == "diabetes":
        hba1c = st.number_input("HbA1c (%)", 3.0, 15.0, 5.5)
        glucose = st.number_input("Glucose (mg/dL)", 50.0, 300.0, 100.0)
    else:
        bp = st.number_input("Blood Pressure", 80.0, 200.0, 120.0)
        cholesterol = st.number_input("Cholesterol", 100.0, 350.0, 180.0)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if st.button("Run Risk Assessment", use_container_width=True):
    payload = (
        {"age": age, "bmi": bmi, "hba1c": hba1c, "glucose": glucose}
        if disease == "diabetes"
        else {"age": age, "bmi": bmi, "heart_bp": bp, "cholesterol": cholesterol}
    )

    response = requests.post(
        f"http://127.0.0.1:8000/predict_single?disease={disease}",
        json=payload
    )

    if response.status_code == 200:
        st.session_state.result = response.json()
        st.session_state.risk_score = float(st.session_state.result["risk_score"])
        st.session_state.pdf_bytes = None
    else:
        st.error("Backend error")

# --------------------------------------------------
# RESULTS + PDF
# --------------------------------------------------
if st.session_state.result:
    st.markdown("---")

    confidence = st.session_state.result["confidence_score"]

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Risk Score", f"{st.session_state.risk_score:.2f}")

    with c2:
        st.metric("Clinical Risk Level", st.session_state.result["decision"])

    with c3:
        st.metric("Prediction Confidence", f"{confidence * 100:.0f}%")


    # Create chart
    chart_path = "clinical_chart.png"
    fig, ax = plt.subplots()

    if disease == "diabetes":
        labels = ["Age", "BMI", "HbA1c", "Glucose"]
        values = [age, bmi, hba1c, glucose]
    else:
        labels = ["Age", "BMI", "BP", "Cholesterol"]
        values = [age, bmi, bp, cholesterol]

    ax.bar(labels, values, color="#0f4c81")
    plt.tight_layout()
    fig.savefig(chart_path)
    plt.close(fig)

    if st.button("📄 Generate PDF"):
        patient_data = dict(zip(labels, values))
        pdf_path = "Clinical_AI_Risk_Report.pdf"

        generate_pdf_report(
            pdf_path,
            disease,
            patient_data,
            st.session_state.risk_score,
            st.session_state.result["decision"],
            chart_path
        )

        with open(pdf_path, "rb") as f:
            st.session_state.pdf_bytes = f.read()

        st.success("PDF generated")

    if st.session_state.pdf_bytes:
        st.download_button(
            "⬇️ Download PDF",
            st.session_state.pdf_bytes,
            file_name="Clinical_AI_Risk_Report.pdf",
            mime="application/pdf"
        )

    if os.path.exists(chart_path):
        os.remove(chart_path)
