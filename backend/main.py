from fastapi import FastAPI
from pydantic import BaseModel
import torch
import os

from .model_agent import DiseaseRiskModel
from .decision_agent import interpret_risk

app = FastAPI(title="Agentic AI Disease Risk Prediction Backend")

# --------------------------------------------------
# Input schema
# --------------------------------------------------
class PatientInput(BaseModel):
    age: float
    bmi: float
    hba1c: float | None = None
    glucose: float | None = None
    heart_bp: float | None = None
    cholesterol: float | None = None

# --------------------------------------------------
# Root check
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "Backend is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict_single")
def predict_single(patient: PatientInput, disease: str = "diabetes"):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, f"{disease}_model.pt")

    if not os.path.exists(model_path):
        return {
            "error": f"Model file not found: {model_path}",
            "risk_score": 0.0,
            "confidence_score": 0.0,
            "decision": "Unavailable"
        }

    model = DiseaseRiskModel()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # -----------------------------
    # Input handling (safe defaults)
    # -----------------------------
    if disease == "diabetes":
        X = torch.tensor(
            [[
                patient.age,
                patient.bmi,
                patient.hba1c or 0.0,
                patient.glucose or 0.0
            ]],
            dtype=torch.float32
        )

    elif disease == "heart":
        X = torch.tensor(
            [[
                patient.age,
                patient.bmi,
                patient.heart_bp or 0.0,
                patient.cholesterol or 0.0
            ]],
            dtype=torch.float32
        )
    else:
        return {
            "error": "Unknown disease",
            "risk_score": 0.0,
            "confidence_score": 0.0,
            "decision": "Unavailable"
        }

    # -----------------------------
    # Prediction
    # -----------------------------
    with torch.no_grad():
        risk = model(X).item()

    # Clamp safety
    risk = max(0.0, min(1.0, float(risk)))

    # Confidence estimation
    confidence = round(abs(risk - 0.5) * 2, 2)

    return {
        "disease": disease,
        "risk_score": round(risk, 2),
        "confidence_score": confidence,
        "decision": interpret_risk(risk)
    }
