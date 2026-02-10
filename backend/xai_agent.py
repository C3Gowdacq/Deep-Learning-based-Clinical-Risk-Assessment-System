import shap
import torch
import numpy as np
import matplotlib.pyplot as plt

from .data_agent import load_data
from .preprocess_agent import preprocess
from .model_agent import DiseaseRiskModel


def compute_shap(disease):
    # Load data
    data = load_data()
    X, _ = preprocess(data, disease)

    # Feature names
    if disease == "diabetes":
        feature_names = ["age", "bmi", "hba1c", "glucose"]
    elif disease == "heart":
        feature_names = ["age", "bmi", "heart_bp", "cholesterol"]
    else:
        raise ValueError("Unknown disease")

    # Load trained model
    model = DiseaseRiskModel()
    model.load_state_dict(torch.load(f"backend/{disease}_model.pt"))
    model.eval()

    X_np = X.numpy()

    # Background samples (small subset)
    background = X_np[:3]

    # SHAP explainer
    explainer = shap.KernelExplainer(
        lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy(),
        background
    )

    shap_values = explainer.shap_values(X_np[:1])

    return shap_values[0], feature_names, X_np[:1]
