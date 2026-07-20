"""AML classifier model and config shared between client and server."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    """Hyperparameters that define the model architecture."""
    input_dim: int       = 19
    hidden_dims: list    = field(default_factory=lambda: [64, 32, 16])
    dropout_rate: float  = 0.3


class AMLClassifier(nn.Module):
    """
    MLP for binary fraud classification.

    Architecture: Input(19) -> FC(64)->BN->ReLU->Dropout
                            -> FC(32)->BN->ReLU->Dropout
                            -> FC(16)->BN->ReLU
                            -> FC(1)
    """

    def __init__(self, config: ModelConfig = None):
        super().__init__()
        cfg = config or ModelConfig()

        layers = []
        in_dim = cfg.input_dim
        for i, out_dim in enumerate(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            # no dropout on the last hidden layer
            if i < len(cfg.hidden_dims) - 1:
                layers.append(nn.Dropout(cfg.dropout_rate))
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return fraud probability (sigmoid of logit)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    def get_weights(self) -> dict:
        """Return a copy of the state dict for sending to the server."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict) -> None:
        """Load weights received from the server."""
        self.load_state_dict(weights)


def make_model(config: ModelConfig = None) -> AMLClassifier:
    """Create a fresh AMLClassifier from config."""
    return AMLClassifier(config or ModelConfig())
