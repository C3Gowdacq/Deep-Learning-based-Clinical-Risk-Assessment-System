import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_agent import load_data
from preprocess_agent import preprocess
from model_agent import DiseaseRiskModel

# -----------------------------
# Load & preprocess data
# -----------------------------
data = load_data()
X, y = preprocess(data, "diabetes")

# Convert to numpy for splitting
X = X.numpy()
y = y.numpy()

# -----------------------------
# Train / Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert back to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = DiseaseRiskModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# Training loop
# -----------------------------
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    preds = model(X_train).squeeze()
    loss = criterion(preds, y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4f}")

# -----------------------------
# Evaluation
# -----------------------------
model.eval()
with torch.no_grad():
    test_preds = model(X_test).squeeze()
    test_preds_class = (test_preds >= 0.5).float()
    accuracy = accuracy_score(y_test, test_preds_class)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# Save model
# -----------------------------
torch.save(model.state_dict(), "backend/diabetes_model.pt")
print("Diabetes model trained and saved successfully")
