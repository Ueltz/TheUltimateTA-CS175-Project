import numpy as np
from sklearn.metrics import (
    cohen_kappa_score, mean_squared_error, mean_absolute_error,
    accuracy_score, confusion_matrix,
)
from typing import Dict, Optional, List
import json


def compute_qwk(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return cohen_kappa_score(true_labels, pred_labels, weights='quadratic')


def compute_metrics(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    pred_norm: Optional[np.ndarray] = None,
    true_norm: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    metrics = {}
    metrics['qwk'] = compute_qwk(true_labels, pred_labels)
    metrics['exact_accuracy'] = accuracy_score(true_labels, pred_labels)
    metrics['adjacent_accuracy'] = float(np.mean(np.abs(true_labels - pred_labels) <= 1))

    for label in sorted(set(true_labels) | set(pred_labels)):
        mask = true_labels == label
        if mask.sum() > 0:
            metrics[f'accuracy_class_{label}'] = float(np.mean(pred_labels[mask] == label))

    if pred_norm is not None and true_norm is not None:
        metrics['rmse'] = float(np.sqrt(mean_squared_error(true_norm, pred_norm)))
        metrics['mae'] = float(mean_absolute_error(true_norm, pred_norm))

    return metrics


def compute_per_prompt_qwk(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    prompt_ids: np.ndarray,
) -> Dict[int, float]:
    results = {}
    for pid in sorted(np.unique(prompt_ids)):
        mask = prompt_ids == pid
        if mask.sum() < 2:
            continue
        results[int(pid)] = compute_qwk(true_labels[mask], pred_labels[mask])
    return results


def paired_bootstrap_qwk(
    true_labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    n = len(true_labels)
    qwk_a = compute_qwk(true_labels, preds_a)
    qwk_b = compute_qwk(true_labels, preds_b)
    observed_delta = qwk_a - qwk_b

    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            d = compute_qwk(true_labels[idx], preds_a[idx]) - compute_qwk(true_labels[idx], preds_b[idx])
            deltas.append(d)
        except Exception:
            continue

    deltas = np.array(deltas)
    return {
        'qwk_a': qwk_a,
        'qwk_b': qwk_b,
        'delta': observed_delta,
        'ci_lower': float(np.percentile(deltas, 2.5)),
        'ci_upper': float(np.percentile(deltas, 97.5)),
        'p_value': float(np.mean(deltas < 0) if observed_delta >= 0 else np.mean(deltas > 0)),
    }


def print_confusion_matrix(true_labels, pred_labels, label_names=None):
    labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    if label_names is None:
        label_names = [str(l) for l in labels]
    header = "      " + "  ".join(f"{n:>5}" for n in label_names)
    print(header)
    print("      " + "-" * (len(header) - 6))
    for i, row in enumerate(cm):
        print(f"{label_names[i]:>5} | " + "  ".join(f"{v:>5}" for v in row))


def error_analysis(true_labels, pred_labels) -> Dict[str, float]:
    diffs = pred_labels - true_labels
    abs_diffs = np.abs(diffs)
    return {
        'exact_match': float(np.mean(abs_diffs == 0)),
        'off_by_1': float(np.mean(abs_diffs == 1)),
        'off_by_2': float(np.mean(abs_diffs == 2)),
        'off_by_3_plus': float(np.mean(abs_diffs >= 3)),
        'mean_absolute_error': float(abs_diffs.mean()),
        'bias': float(diffs.mean()),
        'mean_error_on_low_scores': float(
            diffs[true_labels <= np.percentile(true_labels, 25)].mean()
        ) if len(true_labels) > 0 else 0.0,
        'mean_error_on_high_scores': float(
            diffs[true_labels >= np.percentile(true_labels, 75)].mean()
        ) if len(true_labels) > 0 else 0.0,
    }


def full_evaluation_report(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    prompt_ids: Optional[np.ndarray] = None,
    continuous_preds: Optional[np.ndarray] = None,
    system_name: str = "Model",
    save_path: Optional[str] = None,
) -> Dict:
    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT: {system_name}")
    print(f"{'=' * 60}")

    metrics = compute_metrics(true_labels, pred_labels)
    print(f"\nOverall QWK:           {metrics['qwk']:.4f}")
    print(f"Exact accuracy:        {metrics['exact_accuracy']:.4f}")
    print(f"Adjacent accuracy:     {metrics['adjacent_accuracy']:.4f}")

    errors = error_analysis(true_labels, pred_labels)
    print(f"\nError Distribution:")
    print(f"  Exact match:   {errors['exact_match']:.1%}")
    print(f"  Off by 1:      {errors['off_by_1']:.1%}")
    print(f"  Off by 2:      {errors['off_by_2']:.1%}")
    print(f"  Off by 3+:     {errors['off_by_3_plus']:.1%}")
    print(f"  Mean abs error: {errors['mean_absolute_error']:.3f}")
    print(f"  Prediction bias: {errors['bias']:+.3f}")

    per_prompt = {}
    if prompt_ids is not None:
        per_prompt = compute_per_prompt_qwk(true_labels, pred_labels, prompt_ids)
        print(f"\nPer-prompt QWK:")
        for pid, qwk in sorted(per_prompt.items()):
            n = (prompt_ids == pid).sum()
            print(f"  Prompt {pid}: QWK = {qwk:.4f} (n={n})")

    print(f"\nConfusion Matrix (true \\ pred):")
    print_confusion_matrix(true_labels, pred_labels)

    report = {
        'system_name': system_name,
        'overall_metrics': metrics,
        'error_analysis': errors,
        'per_prompt_qwk': per_prompt,
    }

    if save_path:
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        serializable = json.loads(json.dumps(report, default=convert))
        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\nReport saved to {save_path}")

    return report
