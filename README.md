# Federated AML Engine

Implements Federated Averaging (FedAvg) for anti-money laundering fraud detection across three simulated banks. Each bank trains a local model on its own private transaction data, and only model weights are sent to a central aggregator - no raw data is ever shared. The aggregator merges the weights and redistributes an improved global model each round.

Built to understand federated learning in a domain where data privacy is a real constraint. Banks are legally prohibited from sharing customer data across institutions, but money laundering patterns often span multiple banks.

---

## How it works

**FedAvg formula:**

```
w_global = sum_k (n_k / N) * w_k
```

Each bank's contribution to the global model is weighted by its dataset size. A bank with more training samples has proportionally more influence.

**Per round:**
1. The central server distributes the current global weights to all banks
2. Each bank runs local SGD for a fixed number of epochs on its private data
3. Updated weights are sent back to the server (no data, only weights)
4. The server computes the weighted average and installs the new global model
5. The global model is evaluated on a held-out test set

**Data:** Each bank gets a synthetic transaction dataset with ~2% fraud rate. Features include transaction amount, time encoding, velocity counts, cross-border flag, new-recipient flag, round-amount flag, and merchant category (one-hot). Banks have different transaction distributions to simulate real non-IID federated data.

**Model:** Three-layer MLP with batch normalisation and dropout. BCEWithLogitsLoss with upweighted fraud class to handle class imbalance.

---

## Usage

```bash
pip install -r requirements.txt
python simulation_runner.py
```

Output per round:

```
Round  1 | AUROC: 0.8214 | F1: 0.4123 | Recall: 0.5810 | Precision: 0.3201 | Acc: 0.9612
Round  2 | AUROC: 0.8531 | F1: 0.4672 | Recall: 0.6102 | Precision: 0.3784 | Acc: 0.9649
...
```

Configuration is at the top of `simulation_runner.py`:

```python
NUM_BANKS         = 3
NUM_ROUNDS        = 10
LOCAL_EPOCHS      = 5
SAMPLES_PER_BANK  = 10_000
FRAUD_RATE        = 0.02
```

---

## Project structure

```
federated-aml/
├── shared/
│   └── model_definition.py     # MLP architecture shared by all parties
├── client/
│   ├── data_loader.py          # synthetic transaction data generator
│   └── local_trainer.py        # local SGD training loop
├── server/
│   ├── central_aggregator.py   # FedAvg weight averaging
│   └── evaluator.py            # AUROC, F1, precision, recall per round
├── simulation_runner.py        # full federation simulation
└── tests/
    ├── test_fedavg.py
    └── test_data_loader.py
```

---

## Stack

Python 3.10, PyTorch

No external dataset download required. Synthetic data is generated on the fly.
