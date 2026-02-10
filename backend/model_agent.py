import torch
import torch.nn as nn

class DiseaseRiskModel(nn.Module):
    def __init__(self):
        super().__init__()
        # MUST match training-time model
        self.layer = nn.Linear(4, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))
