import torch
from torch import nn


class PlaceholderSERClassifier(nn.Module):
    """Simple classifier used only to keep the pipeline executable."""

    def __init__(self, model_cfg):
        super().__init__()
        input_dim = int(model_cfg["input_dim"])
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        dropout = float(model_cfg.get("dropout", 0.1))
        num_classes = int(model_cfg["num_classes"])

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, waveform):
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform shape [batch, samples], got {tuple(waveform.shape)}")
        return self.net(waveform.float())

