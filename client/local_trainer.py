"""Local trainer - runs SGD on a bank's private data and returns updated weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.model_definition import AMLClassifier, ModelConfig, make_model

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    learning_rate: float      = 1e-3
    local_epochs: int         = 5
    batch_size: int           = 64
    fraud_class_weight: float = 10.0  # upweight fraud since it's only 2% of data
    max_grad_norm: float      = 1.0
    device: str               = "cpu"


@dataclass
class LocalTrainResult:
    bank_id: int
    weights: dict          # updated model weights to send to server
    n_samples: int
    avg_loss: float
    epochs_trained: int
    train_accuracy: float


class LocalTrainer:
    """
    Trains the shared AML model on a bank's local dataset.

    The bank keeps its transaction data private - only the updated
    model weights are returned to the central aggregator.
    """

    def __init__(self, bank_id: int, config: TrainConfig = None):
        self.bank_id = bank_id
        self.config  = config or TrainConfig()
        self.device  = torch.device(self.config.device)

    def train(self, dataloader: DataLoader, global_weights: dict,
              model_config: ModelConfig = None) -> LocalTrainResult:
        """Load global weights, train locally, return updated weights."""
        model = make_model(model_config)
        model.set_weights(global_weights)
        model.to(self.device)
        model.train()

        # pos_weight handles the severe class imbalance (2% fraud)
        pos_weight = torch.tensor([self.config.fraud_class_weight]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        total_loss    = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.config.local_epochs):
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()

                # gradient clipping stops exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()

                total_loss    += loss.item() * len(y_batch)
                preds          = (torch.sigmoid(logits) >= 0.5).float()
                total_correct += (preds == y_batch).sum().item()
                total_samples += len(y_batch)

        avg_loss     = total_loss / max(total_samples, 1)
        train_acc    = total_correct / max(total_samples, 1)
        n_samples    = len(dataloader.dataset)

        log.info("Bank %d: loss=%.4f acc=%.3f n=%d",
                 self.bank_id, avg_loss, train_acc, n_samples)

        return LocalTrainResult(
            bank_id=self.bank_id,
            weights=model.get_weights(),
            n_samples=n_samples,
            avg_loss=avg_loss,
            epochs_trained=self.config.local_epochs,
            train_accuracy=train_acc,
        )
