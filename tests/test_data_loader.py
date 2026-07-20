"""Tests for the synthetic data generator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
from client.data_loader import generate_bank_data, make_dataloader, FEATURE_DIM


class TestBankDataset:
    def test_correct_number_of_samples(self):
        ds = generate_bank_data(bank_id=0, n_samples=500, seed=42)
        assert len(ds) == 500

    def test_feature_dimension(self):
        ds = generate_bank_data(bank_id=0, n_samples=100, seed=1)
        x, y = ds[0]
        assert x.shape == (FEATURE_DIM,)

    def test_labels_are_binary(self):
        ds = generate_bank_data(bank_id=0, n_samples=500, seed=42)
        unique = set(ds.labels.numpy().tolist())
        assert unique <= {0.0, 1.0}

    def test_fraud_rate_approximately_correct(self):
        ds = generate_bank_data(bank_id=0, n_samples=5000, fraud_rate=0.05, seed=99)
        assert 0.03 <= ds.fraud_rate <= 0.07

    def test_n_fraud_property(self):
        ds = generate_bank_data(bank_id=0, n_samples=1000, fraud_rate=0.02, seed=7)
        assert ds.n_fraud + ds.n_legit == len(ds)

    def test_different_banks_have_different_data(self):
        ds0 = generate_bank_data(bank_id=0, n_samples=200, seed=0)
        ds1 = generate_bank_data(bank_id=1, n_samples=200, seed=0)
        # mean amounts differ because of bank-specific bias
        assert not torch.allclose(ds0.features[:, 0].mean(), ds1.features[:, 0].mean())

    def test_seed_reproducibility(self):
        ds_a = generate_bank_data(bank_id=0, n_samples=100, seed=42)
        ds_b = generate_bank_data(bank_id=0, n_samples=100, seed=42)
        assert torch.allclose(ds_a.features, ds_b.features)

    def test_dataloader_batch_shape(self):
        ds = generate_bank_data(bank_id=0, n_samples=128, seed=5)
        dl = make_dataloader(ds, batch_size=32, shuffle=False)
        X, y = next(iter(dl))
        assert X.shape == (32, FEATURE_DIM)
        assert y.shape == (32,)

    def test_features_are_float32(self):
        ds = generate_bank_data(bank_id=0, n_samples=100, seed=3)
        assert ds.features.dtype == torch.float32

    def test_merchant_one_hot_sums_to_one(self):
        ds = generate_bank_data(bank_id=0, n_samples=200, seed=10)
        merchant_cols = ds.features[:, 11:19]
        row_sums = merchant_cols.sum(dim=1)
        # each row should have exactly one merchant category
        assert torch.all(row_sums == 1.0)
