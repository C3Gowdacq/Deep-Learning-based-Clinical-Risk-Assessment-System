import torch
import pandas as pd
import matplotlib.pyplot as plt

from data_agent import load_data
from preprocess_agent import preprocess
from model_agent import DiseaseRiskModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data = load_data()
results = []

# ==================================================
# DIABETES EVALUATION
# ==================================================
X_d, y_d = preprocess(data, "diabetes")

diabetes_model = DiseaseRiskModel()
diabetes_model.load_state_dict(
    torch.load("backend/diabetes_model.pt", map_location="cpu")
)
diabetes_model.eval()

with torch.no_grad():
    probs_d = diabetes_model(X_d).squeeze()
    preds_d = (probs_d >= 0.5).int()

# Metrics
acc_d = accuracy_score(y_d, preds_d)
prec_d = precision_score(y_d, preds_d)
rec_d = recall_score(y_d, preds_d)
f1_d = f1_score(y_d, preds_d)
roc_d = roc_auc_score(y_d, probs_d)

results.append({
    "Disease": "Diabetes",
    "Model": "Neural Network (MLP)",
    "Accuracy": acc_d,
    "Precision": prec_d,
    "Recall": rec_d,
    "F1-Score": f1_d,
    "ROC-AUC": roc_d
})

# Confusion Matrix
cm_d = confusion_matrix(y_d, preds_d)
disp_d = ConfusionMatrixDisplay(
    confusion_matrix=cm_d,
    display_labels=["No Diabetes", "Diabetes"]
)
disp_d.plot(cmap="Blues")
plt.title("Diabetes – Confusion Matrix")
plt.savefig("confusion_diabetes.png", bbox_inches="tight")
plt.close()

# ROC Curve
fpr_d, tpr_d, _ = roc_curve(y_d, probs_d)
plt.plot(fpr_d, tpr_d, label=f"ROC AUC = {roc_d:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Diabetes – ROC Curve")
plt.legend()
plt.savefig("roc_diabetes.png", bbox_inches="tight")
plt.close()

# ==================================================
# HEART DISEASE EVALUATION
# ==================================================
X_h, y_h = preprocess(data, "heart")

heart_model = DiseaseRiskModel()
heart_model.load_state_dict(
    torch.load("backend/heart_model.pt", map_location="cpu")
)
heart_model.eval()

with torch.no_grad():
    probs_h = heart_model(X_h).squeeze()
    preds_h = (probs_h >= 0.5).int()

# Metrics
acc_h = accuracy_score(y_h, preds_h)
prec_h = precision_score(y_h, preds_h)
rec_h = recall_score(y_h, preds_h)
f1_h = f1_score(y_h, preds_h)
roc_h = roc_auc_score(y_h, probs_h)

results.append({
    "Disease": "Heart Disease",
    "Model": "Neural Network (MLP)",
    "Accuracy": acc_h,
    "Precision": prec_h,
    "Recall": rec_h,
    "F1-Score": f1_h,
    "ROC-AUC": roc_h
})

# Confusion Matrix
cm_h = confusion_matrix(y_h, preds_h)
disp_h = ConfusionMatrixDisplay(
    confusion_matrix=cm_h,
    display_labels=["No Heart Disease", "Heart Disease"]
)
disp_h.plot(cmap="Reds")
plt.title("Heart Disease – Confusion Matrix")
plt.savefig("confusion_heart.png", bbox_inches="tight")
plt.close()

# ROC Curve
fpr_h, tpr_h, _ = roc_curve(y_h, probs_h)
plt.plot(fpr_h, tpr_h, label=f"ROC AUC = {roc_h:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Heart Disease – ROC Curve")
plt.legend()
plt.savefig("roc_heart.png", bbox_inches="tight")
plt.close()

# ==================================================
# SAVE & DISPLAY RESULTS
# ==================================================
df_results = pd.DataFrame(results)
df_results.to_csv("final_results.csv", index=False)

print("\nFINAL EVALUATION RESULTS")
print(df_results)
print("\nSaved files:")
print("- final_results.csv")
print("- confusion_diabetes.png")
print("- confusion_heart.png")
print("- roc_diabetes.png")
print("- roc_heart.png")
