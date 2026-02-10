import shap
import matplotlib.pyplot as plt
from .xai_agent import compute_shap

# Choose disease
DISEASE = "diabetes"   # or "heart"

shap_values, feature_names, X_sample = compute_shap(DISEASE)

# Convert to readable format
shap_values = shap_values[0]

# Bar plot (global importance for one sample)
plt.figure()
shap.bar_plot(
    shap_values,
    feature_names=feature_names,
    show=False
)
plt.title(f"SHAP Feature Importance ({DISEASE.capitalize()})")
plt.tight_layout()
plt.savefig(f"backend/shap_{DISEASE}_bar.png")
plt.close()

print("SHAP bar plot saved")
