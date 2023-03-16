import torch
import torch.nn as nn

#### Input shape:  (n, num_feature)
#### Output shape: (n, 1)
class Simple_MLP(nn.Module):
    def __init__(self, num_feature: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_feature, 64), 
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return torch.sigmoid(self.layers(x))