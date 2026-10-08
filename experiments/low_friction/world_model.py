import torch
from torch import nn


class DynamicsTransformer(nn.Module):
    def __init__(self, history_length=10, width=64, layers=2, heads=4):
        super().__init__()
        self.config = dict(history_length=history_length, width=width, layers=layers, heads=heads)
        self.input = nn.Linear(7, width)
        self.position = nn.Parameter(torch.zeros(1, history_length, width))
        nn.init.normal_(self.position, std=0.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width*2,
            dropout=0.0, activation='gelu', batch_first=True), num_layers=layers,
            enable_nested_tensor=False)
        self.output = nn.Sequential(nn.Linear(width+2, width), nn.GELU(), nn.Linear(width, 8))

    def forward(self, history, action):
        encoded = self.encoder(self.input(history) + self.position)
        return self.output(torch.cat([encoded[:, -1], action], dim=-1))
