"""Tests for the FedAvg aggregation logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from unittest.mock import MagicMock
from server.central_aggregator import CentralAggregator
from shared.model_definition import ModelConfig


def _make_weights(val: float, config: ModelConfig) -> dict:
    """Create a state dict where every parameter is filled with val."""
    from shared.model_definition import make_model
    m = make_model(config)
    return {k: torch.full_like(v, val) for k, v in m.state_dict().items()}


def _make_result(bank_id: int, val: float, n_samples: int, config: ModelConfig):
    from client.local_trainer import LocalTrainResult
    return LocalTrainResult(
        bank_id=bank_id,
        weights=_make_weights(val, config),
        n_samples=n_samples,
        avg_loss=0.5,
        epochs_trained=5,
        train_accuracy=0.9,
    )


@pytest.fixture
def config():
    return ModelConfig()


@pytest.fixture
def aggregator(config):
    return CentralAggregator(config)


class TestFedAvg:
    def test_round_counter_starts_at_zero(self, aggregator):
        assert aggregator.round_num == 0

    def test_round_counter_increments(self, aggregator, config):
        r = _make_result(0, 1.0, 100, config)
        aggregator.aggregate([r])
        assert aggregator.round_num == 1

    def test_equal_banks_gives_simple_average(self, aggregator, config):
        # two banks with equal samples: result should be average of their weights
        r0 = _make_result(0, 2.0, 100, config)
        r1 = _make_result(1, 4.0, 100, config)
        aggregator.aggregate([r0, r1])
        weights = aggregator.global_weights
        for v in weights.values():
            assert torch.allclose(v, torch.tensor(3.0), atol=1e-5)

    def test_large_bank_dominates(self, aggregator, config):
        # bank 0 has 9x more data so global weights should be closer to its values
        r0 = _make_result(0, 0.0, 900, config)
        r1 = _make_result(1, 1.0, 100, config)
        aggregator.aggregate([r0, r1])
        weights = aggregator.global_weights
        for v in weights.values():
            assert torch.allclose(v, torch.tensor(0.1), atol=1e-5)

    def test_single_bank(self, aggregator, config):
        r = _make_result(0, 7.0, 500, config)
        aggregator.aggregate([r])
        weights = aggregator.global_weights
        for v in weights.values():
            assert torch.allclose(v, torch.tensor(7.0), atol=1e-5)

    def test_empty_results_raises(self, aggregator):
        with pytest.raises(ValueError):
            aggregator.aggregate([])

    def test_summary_has_correct_banks(self, aggregator, config):
        r0 = _make_result(0, 1.0, 100, config)
        r1 = _make_result(2, 2.0, 100, config)
        summary = aggregator.aggregate([r0, r1])
        assert 0 in summary.participating_banks
        assert 2 in summary.participating_banks

    def test_summary_total_samples(self, aggregator, config):
        r0 = _make_result(0, 1.0, 300, config)
        r1 = _make_result(1, 2.0, 700, config)
        summary = aggregator.aggregate([r0, r1])
        assert summary.total_samples == 1000

    def test_model_is_accessible_after_aggregation(self, aggregator, config):
        r = _make_result(0, 1.0, 100, config)
        aggregator.aggregate([r])
        model = aggregator.global_model
        assert model is not None
