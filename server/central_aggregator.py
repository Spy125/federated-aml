"""Central aggregator - implements FedAvg to merge client model updates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from client.local_trainer import LocalTrainResult
from shared.model_definition import AMLClassifier, ModelConfig, make_model

log = logging.getLogger(__name__)


@dataclass
class RoundSummary:
    round_num: int
    participating_banks: list[int]
    total_samples: int
    avg_loss: float


class CentralAggregator:
    """
    Federated server that implements FedAvg (McMahan et al., 2017).

    FedAvg formula:
        w_global = sum_k  (n_k / N) * w_k

    where n_k is the number of training samples at client k,
    and N = sum_k n_k is the total across all participating clients.

    No raw data is ever sent to the server - only model weight tensors.
    """

    def __init__(self, model_config: ModelConfig = None):
        self._config       = model_config or ModelConfig()
        self._global_model = make_model(self._config)
        self._round        = 0

    @property
    def global_model(self) -> AMLClassifier:
        return self._global_model

    @property
    def global_weights(self) -> dict:
        return self._global_model.get_weights()

    @property
    def round_num(self) -> int:
        return self._round

    def aggregate(self, client_results: list[LocalTrainResult]) -> RoundSummary:
        """Average client weights weighted by number of samples (FedAvg)."""
        if not client_results:
            raise ValueError("No client results to aggregate")

        self._round += 1
        total_samples = sum(r.n_samples for r in client_results)

        # weighted average of each parameter tensor
        avg_weights: dict = {}
        for result in client_results:
            weight = result.n_samples / total_samples
            for key, tensor in result.weights.items():
                # Integer buffers (for example BatchNorm's num_batches_tracked)
                # cannot hold a weighted mean, and accumulating a float into a
                # zeros_like copy of them raises a dtype error. Carry the first
                # client's value through instead of averaging.
                if not torch.is_floating_point(tensor):
                    if key not in avg_weights:
                        avg_weights[key] = tensor.clone()
                    continue
                if key not in avg_weights:
                    avg_weights[key] = torch.zeros_like(tensor)
                avg_weights[key] += weight * tensor

        self._global_model.set_weights(avg_weights)

        summary = RoundSummary(
            round_num=self._round,
            participating_banks=[r.bank_id for r in client_results],
            total_samples=total_samples,
            avg_loss=sum(r.avg_loss for r in client_results) / len(client_results),
        )
        log.info("Round %d: aggregated %d banks, %d total samples",
                 self._round, len(client_results), total_samples)
        return summary

    def save(self, path: str) -> None:
        """Save the global model to disk."""
        torch.save({
            "round": self._round,
            "state_dict": self._global_model.state_dict(),
            "config": self._config,
        }, path)
        log.info("Global model saved to %s (round %d)", path, self._round)

    def load(self, path: str) -> None:
        """Load a saved global model from disk."""
        checkpoint = torch.load(path, map_location="cpu")
        self._round = checkpoint["round"]
        self._global_model.load_state_dict(checkpoint["state_dict"])
        log.info("Loaded global model from %s (round %d)", path, self._round)
