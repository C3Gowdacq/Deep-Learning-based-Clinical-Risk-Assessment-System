import torch

def preprocess(data, disease):
    if disease == "diabetes":
        X = data[["age", "bmi", "hba1c", "glucose"]]
        y = data["diabetes_label"]

    elif disease == "heart":
        X = data[["age", "bmi", "systolic_bp", "cholesterol"]]
        y = data["heart_disease_label"]

    else:
        raise ValueError("Unknown disease type")

    X = torch.tensor(X.values, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)

    return X, y
