from torch import nn
import torch

class ParticleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,9),
        )
    def forward(self,inputs):
        return self.network(inputs)

class NeighborhoodMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        self.predictor = nn.Sequential(
            nn.Linear(21 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )

    def forward(self, inputs, neighbors, mask):
        encoded = self.neighbor_encoder(neighbors)
        encoded = encoded * mask.unsqueeze(-1)
        pooled = encoded.sum(dim=1)
        combined = torch.cat([inputs, pooled], dim=1)
        return self.predictor(combined)