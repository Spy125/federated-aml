"""Global model evaluator - computes AML detection metrics on held-out data."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader

from shared.model_definition import AMLClassifier

log = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    round_num: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    n_samples: int
    n_fraud: int
    n_legit: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def false_positive_rate(self) -> float:
        return self.fp / max(self.fp + self.tn, 1)

    @property
    def detection_rate(self) -> float:
        """Recall alias - fraction of true fraud cases caught."""
        return self.recall

    def summary_line(self) -> str:
        return (
            f"Round {self.round_num:>3} | "
            f"AUROC: {self.auroc:.4f} | "
            f"F1: {self.f1:.4f} | "
            f"Recall: {self.recall:.4f} | "
            f"Precision: {self.precision:.4f} | "
            f"Acc: {self.accuracy:.4f}"
        )


def balanced_threshold(pos_weight: float) -> float:
    """Decision threshold that offsets training with a weighted positive class.

    BCEWithLogitsLoss(pos_weight=w) multiplies the loss on positive examples by
    w, and the optimum of that weighted objective inflates the predicted odds by
    the same factor: odds' = w * odds. Probabilities from such a model are
    therefore not calibrated, and comparing them against 0.5 classifies far too
    much of the data as positive. Undoing the inflation and asking where the
    corrected probability reaches 0.5 gives a cutoff of w / (1 + w).

    With the default w = 10 this is 0.909 rather than 0.5.
    """
    if pos_weight <= 0:
        return 0.5
    return pos_weight / (1.0 + pos_weight)


class Evaluator:
    """Runs the global model on the held-out test set after each round."""

    def __init__(self, threshold: float = 0.5, device: str = "cpu"):
        self._threshold = threshold
        self.device = torch.device(device)
        self._history: list[EvalMetrics] = []

    @property
    def history(self) -> list[EvalMetrics]:
        return self._history

    @property
    def threshold(self) -> float:
        return self._threshold

    def _collect(self, model: AMLClassifier, dataloader: DataLoader):
        """Run the model over a loader and return (probabilities, labels)."""
        model.eval()
        model.to(self.device)
        probs: list[float] = []
        labels: list[float] = []
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                batch_probs = model.predict_proba(X_batch)
                probs.extend(batch_probs.cpu().numpy().tolist())
                labels.extend(y_batch.numpy().tolist())
        return np.array(probs), np.array(labels)

    def tune_threshold(self, model: AMLClassifier, dataloader: DataLoader) -> float:
        """Pick the decision threshold that maximises F1 on a validation set.

        Training upweights the fraud class, so the model's probabilities are not
        calibrated and 0.5 is not a meaningful cutoff. The size of that shift
        also grows as training proceeds, which means no fixed threshold serves
        every round. Selecting it on a validation split keeps the held-out test
        set untouched for reporting.
        """
        probs, labels = self._collect(model, dataloader)
        if labels.sum() == 0 or len(labels) == 0:
            return self._threshold

        # Candidate cutoffs drawn from the upper range of observed scores, where
        # a heavily imbalanced model places its decision boundary.
        candidates = np.unique(np.quantile(probs, np.linspace(0.5, 0.9995, 300)))
        best_threshold, best_f1 = self._threshold, -1.0
        for t in candidates:
            preds = probs >= t
            tp = int((preds & (labels == 1)).sum())
            fp = int((preds & (labels == 0)).sum())
            fn = int(((~preds) & (labels == 1)).sum())
            precision = tp / max(tp + fp, 1)
            recall    = tp / max(tp + fn, 1)
            f1        = 2 * precision * recall / max(precision + recall, 1e-9)
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(t)

        self._threshold = best_threshold
        return best_threshold

    def evaluate(self, model: AMLClassifier, dataloader: DataLoader,
                 round_num: int = 0) -> EvalMetrics:
        """Run the model on all batches and compute metrics."""
        probs_arr, labels_arr = self._collect(model, dataloader)
        preds_arr = (probs_arr >= self._threshold).astype(float)

        tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
        fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
        tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())
        fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())

        accuracy  = (tp + tn) / max(len(labels_arr), 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-9)
        auroc     = self._compute_auroc(labels_arr, probs_arr)

        metrics = EvalMetrics(
            round_num=round_num,
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            auroc=round(auroc, 4),
            n_samples=len(labels_arr),
            n_fraud=int(labels_arr.sum()),
            n_legit=int((labels_arr == 0).sum()),
            tp=tp, fp=fp, tn=tn, fn=fn,
        )
        self._history.append(metrics)
        log.info(metrics.summary_line())
        return metrics

    def print_history(self) -> None:
        if not self._history:
            print("No evaluation history.")
            return
        print("\n" + "=" * 80)
        print("FEDERATED TRAINING HISTORY")
        print("=" * 80)
        for m in self._history:
            print(m.summary_line())
        best = max(self._history, key=lambda m: m.auroc)
        print(f"\nBest AUROC: {best.auroc:.4f} at round {best.round_num}")
        print("=" * 80 + "\n")

    @staticmethod
    def _compute_auroc(labels: np.ndarray, probs: np.ndarray) -> float:
        """AUROC via trapezoidal rule - no sklearn needed."""
        if labels.sum() == 0 or labels.sum() == len(labels):
            return 0.5  # degenerate case

        order = np.argsort(-probs)
        labels_sorted = labels[order]

        n_pos = labels.sum()
        n_neg = len(labels) - n_pos

        tp_count = 0
        fp_count = 0
        prev_tp  = 0
        auc      = 0.0

        for label in labels_sorted:
            if label == 1:
                tp_count += 1
            else:
                fp_count += 1
                auc += (tp_count + prev_tp) / 2.0 / n_pos
                prev_tp = tp_count

        return auc / max(n_neg, 1)
