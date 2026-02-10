import torch
import torch.nn as nn
import torch.optim as optim

from data_agent import load_data
from preprocess_agent import preprocess
from model_agent import DiseaseRiskModel

# Load data
data = load_data()
X, y = preprocess(data)

# Initialize model
model = DiseaseRiskModel()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 200

for epoch in range(epochs):
    optimizer.zero_grad()

    predictions = model(X).squeeze()
    loss = criterion(predictions, y)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save trained model
torch.save(model.state_dict(), "backend/trained_model.pt")

print("Training complete. Model saved.")
