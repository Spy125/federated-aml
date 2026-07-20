"""
Federated AML Engine - full simulation runner.

Orchestrates NUM_BANKS simulated banks for NUM_ROUNDS of federated training
using FedAvg, with evaluation after every round.

Run:
    python simulation_runner.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from client.data_loader import generate_bank_data, make_dataloader
from client.local_trainer import LocalTrainer, TrainConfig
from server.central_aggregator import CentralAggregator
from server.evaluator import Evaluator
from shared.model_definition import ModelConfig

NUM_BANKS        = 3
NUM_ROUNDS       = 10
LOCAL_EPOCHS     = 5
SAMPLES_PER_BANK = 10_000
TEST_SAMPLES     = 2_000
FRAUD_RATE       = 0.02
BATCH_SIZE       = 64
LEARNING_RATE    = 1e-3
MODEL_SAVE_PATH  = "federated_aml_model.pt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("simulation")


def run_simulation():
    print("\n" + "=" * 70)
    print("  FEDERATED AML ENGINE - Privacy-Preserving Fraud Detection")
    print(f"  Banks: {NUM_BANKS}  |  Rounds: {NUM_ROUNDS}  |  "
          f"Samples/bank: {SAMPLES_PER_BANK:,}  |  Fraud rate: {FRAUD_RATE:.1%}")
    print("=" * 70 + "\n")

    # generate a private dataset for each bank (never shared)
    print("Generating synthetic transaction data...")
    bank_datasets = []
    bank_dataloaders = []
    for bank_id in range(NUM_BANKS):
        ds = generate_bank_data(
            bank_id=bank_id,
            n_samples=SAMPLES_PER_BANK,
            fraud_rate=FRAUD_RATE,
            seed=bank_id * 100,
        )
        dl = make_dataloader(ds, batch_size=BATCH_SIZE, shuffle=True)
        bank_datasets.append(ds)
        bank_dataloaders.append(dl)
        print(f"  Bank {bank_id}: {len(ds):,} transactions | "
              f"Fraud: {ds.n_fraud} ({ds.fraud_rate:.1%})")

    # shared held-out test set (simulates a regulatory evaluation set)
    print("\nGenerating held-out test set...")
    test_ds = generate_bank_data(bank_id=99, n_samples=TEST_SAMPLES,
                                  fraud_rate=FRAUD_RATE, seed=9999)
    test_dl = make_dataloader(test_ds, batch_size=256, shuffle=False)
    print(f"  Test set: {len(test_ds):,} transactions | Fraud: {test_ds.n_fraud}\n")

    # set up federation
    model_config = ModelConfig()
    aggregator   = CentralAggregator(model_config)
    evaluator    = Evaluator(threshold=0.5)
    train_config = TrainConfig(learning_rate=LEARNING_RATE,
                               local_epochs=LOCAL_EPOCHS, batch_size=BATCH_SIZE)
    trainers = [LocalTrainer(bank_id=i, config=train_config) for i in range(NUM_BANKS)]

    # baseline before training
    print("Evaluating untrained baseline...")
    baseline = evaluator.evaluate(aggregator.global_model, test_dl, round_num=0)
    print(f"  {baseline.summary_line()}\n")

    print(f"Starting {NUM_ROUNDS} rounds of federated training...\n")
    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"-- Round {round_num}/{NUM_ROUNDS} " + "-" * 40)

        client_results = []
        global_weights = aggregator.global_weights

        for bank_id, (trainer, dataloader) in enumerate(zip(trainers, bank_dataloaders)):
            result = trainer.train(dataloader=dataloader,
                                   global_weights=global_weights,
                                   model_config=model_config)
            client_results.append(result)
            print(f"  Bank {bank_id}: loss={result.avg_loss:.4f} | "
                  f"acc={result.train_accuracy:.3f} | n={result.n_samples:,}")

        aggregator.aggregate(client_results)
        metrics = evaluator.evaluate(aggregator.global_model, test_dl, round_num=round_num)
        print(f"  -> {metrics.summary_line()}\n")

    evaluator.print_history()
    aggregator.save(MODEL_SAVE_PATH)
    print(f"Final global model saved -> {MODEL_SAVE_PATH}")

    best_auroc = max(evaluator.history, key=lambda m: m.auroc)
    best_f1    = max(evaluator.history, key=lambda m: m.f1)
    print(f"\nBest AUROC: {best_auroc.auroc:.4f} at round {best_auroc.round_num}")
    print(f"Best F1:    {best_f1.f1:.4f} at round {best_f1.round_num}")


if __name__ == "__main__":
    run_simulation()
