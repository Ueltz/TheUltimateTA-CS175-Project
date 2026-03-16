import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize
from typing import Tuple


def optimize_thresholds_qwk(
    true_labels: np.ndarray,
    continuous_preds: np.ndarray,
    num_classes: int = 6,
    resolution: float = 0.01,
) -> Tuple[np.ndarray, float]:
    label_min = true_labels.min()
    thresholds = np.array([label_min + i + 0.5 for i in range(num_classes - 1)])

    best_qwk = -1.0
    best_thresholds = thresholds.copy()

    for round_num in range(5):
        improved = False
        for k in range(len(thresholds)):
            search_range = np.arange(
                thresholds[k] - 1.0,
                thresholds[k] + 1.0 + resolution,
                resolution,
            )
            for candidate in search_range:
                trial = thresholds.copy()
                trial[k] = candidate
                trial = np.sort(trial)

                discrete = np.digitize(continuous_preds, trial) + int(label_min)
                discrete = np.clip(discrete, true_labels.min(), true_labels.max())

                qwk = cohen_kappa_score(true_labels, discrete, weights='quadratic')
                if qwk > best_qwk:
                    best_qwk = qwk
                    best_thresholds = trial.copy()
                    thresholds = trial.copy()
                    improved = True

        if not improved:
            break

    return best_thresholds, best_qwk


def apply_thresholds(
    continuous_preds: np.ndarray,
    thresholds: np.ndarray,
    min_label: int = 0,
    max_label: int = 5,
) -> np.ndarray:
    discrete = np.digitize(continuous_preds, thresholds) + min_label
    return np.clip(discrete, min_label, max_label)


def find_optimal_temperature(
    logits: np.ndarray,
    true_labels: np.ndarray,
    num_classes: int = 6,
) -> float:
    from scipy.special import expit

    def nll_loss(T):
        scaled = logits / T[0]
        total_nll = 0.0
        count = 0
        for k in range(num_classes - 1):
            mask = true_labels >= k
            if mask.sum() == 0:
                continue
            target = (true_labels[mask] > k).astype(float)
            p = expit(scaled[mask, k])
            p = np.clip(p, 1e-7, 1 - 1e-7)
            nll = -(target * np.log(p) + (1 - target) * np.log(1 - p)).mean()
            total_nll += nll
            count += 1
        return total_nll / max(count, 1)

    result = minimize(nll_loss, x0=[1.0], bounds=[(0.01, 10.0)], method='L-BFGS-B')
    return float(result.x[0])
