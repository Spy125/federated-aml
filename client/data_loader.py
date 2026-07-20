"""Synthetic transaction data generator - one dataset per simulated bank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Feature layout (19 dimensions total):
# [0]   log_amount       - log(transaction amount + 1)
# [1]   hour_sin         - sin encoding of hour_of_day
# [2]   hour_cos         - cos encoding of hour_of_day
# [3]   day_sin          - sin encoding of day_of_week
# [4]   day_cos          - cos encoding of day_of_week
# [5]   velocity_1h      - tx count in past hour (normalised /10)
# [6]   velocity_24h     - tx count in past 24h (normalised /50)
# [7]   cross_border     - 0/1 flag
# [8]   new_recipient    - 0/1 flag
# [9]   round_amount     - 0/1 flag (structuring signal)
# [10]  amount_deviation - z-score vs account mean (clipped)
# [11-18] merchant_category - one-hot (8 categories)

FEATURE_DIM         = 19
MERCHANT_CATEGORIES = 8


@dataclass
class BankDataset(Dataset):
    """Synthetic transaction dataset for a single bank."""

    features: torch.Tensor  # (N, FEATURE_DIM) float32
    labels: torch.Tensor    # (N,) float32  0=legit, 1=fraud
    bank_id: int

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    @property
    def n_fraud(self) -> int:
        return int(self.labels.sum().item())

    @property
    def n_legit(self) -> int:
        return len(self) - self.n_fraud

    @property
    def fraud_rate(self) -> float:
        return self.n_fraud / max(len(self), 1)


def generate_bank_data(
    bank_id: int,
    n_samples: int = 10_000,
    fraud_rate: float = 0.02,
    seed: int = None,
) -> BankDataset:
    """
    Generate synthetic AML transaction data for one bank.

    Each bank gets slightly different transaction distributions to simulate
    real-world heterogeneity (non-IID data) across federated clients.
    """
    rng = np.random.default_rng(seed if seed is not None else bank_id * 42)

    n_fraud = max(1, int(n_samples * fraud_rate))
    n_legit = n_samples - n_fraud

    # bank-specific biases so each bank's data looks different
    amount_mean   = 500.0 + bank_id * 200.0
    velocity_bias = bank_id * 0.5

    legit = _gen_transactions(rng, n_legit, is_fraud=False,
                               amount_mean=amount_mean, velocity_bias=velocity_bias)
    fraud = _gen_transactions(rng, n_fraud, is_fraud=True,
                               amount_mean=amount_mean, velocity_bias=velocity_bias)

    X = np.vstack([legit, fraud]).astype(np.float32)
    y = np.array([0.0] * n_legit + [1.0] * n_fraud, dtype=np.float32)

    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]

    return BankDataset(
        features=torch.from_numpy(X),
        labels=torch.from_numpy(y),
        bank_id=bank_id,
    )


def make_dataloader(dataset: BankDataset, batch_size: int = 64,
                    shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _gen_transactions(
    rng: np.random.Generator,
    n: int,
    is_fraud: bool,
    amount_mean: float,
    velocity_bias: float,
) -> np.ndarray:
    rows = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    if is_fraud:
        # fraud: round amounts, cross-border, new recipients, night-time hours
        amounts      = rng.lognormal(mean=np.log(amount_mean * 0.7), sigma=1.2, size=n)
        hours        = rng.choice(list(range(0, 6)) + list(range(22, 24)), size=n)
        days         = rng.integers(0, 7, size=n)
        vel_1h       = rng.poisson(lam=4 + velocity_bias, size=n)
        vel_24h      = rng.poisson(lam=15 + velocity_bias, size=n)
        cross_border = rng.binomial(1, 0.7, size=n)
        new_recip    = rng.binomial(1, 0.8, size=n)
        round_amt    = rng.binomial(1, 0.6, size=n)
        deviation    = rng.normal(loc=2.5, scale=1.0, size=n)
        merchants    = rng.integers(0, MERCHANT_CATEGORIES, size=n)
    else:
        # legit: business hours, known recipients, varied amounts
        amounts      = rng.lognormal(mean=np.log(amount_mean), sigma=0.8, size=n)
        hours        = rng.integers(8, 20, size=n)
        days         = rng.integers(0, 7, size=n)
        vel_1h       = rng.poisson(lam=1 + velocity_bias, size=n)
        vel_24h      = rng.poisson(lam=5 + velocity_bias, size=n)
        cross_border = rng.binomial(1, 0.1, size=n)
        new_recip    = rng.binomial(1, 0.1, size=n)
        round_amt    = rng.binomial(1, 0.15, size=n)
        deviation    = rng.normal(loc=0.0, scale=0.8, size=n)
        merchants    = rng.integers(0, MERCHANT_CATEGORIES, size=n)

    rows[:, 0]  = np.log1p(amounts)
    rows[:, 1]  = np.sin(2 * np.pi * hours / 24)
    rows[:, 2]  = np.cos(2 * np.pi * hours / 24)
    rows[:, 3]  = np.sin(2 * np.pi * days / 7)
    rows[:, 4]  = np.cos(2 * np.pi * days / 7)
    rows[:, 5]  = np.clip(vel_1h / 10.0, 0, 1)
    rows[:, 6]  = np.clip(vel_24h / 50.0, 0, 1)
    rows[:, 7]  = cross_border
    rows[:, 8]  = new_recip
    rows[:, 9]  = round_amt
    rows[:, 10] = np.clip(deviation, -3.0, 3.0) / 3.0

    # one-hot merchant category (features 11-18)
    for i, cat in enumerate(merchants):
        rows[i, 11 + cat] = 1.0

    return rows
