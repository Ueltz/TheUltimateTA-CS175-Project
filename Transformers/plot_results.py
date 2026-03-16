import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix, cohen_kappa_score

FIGURES_DIR = './figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

ZS_HISTORY = './outputs/graphs/training_history.json'
ZS_EVAL = './outputs/graphs/evaluation_data.json'
ZS_REPORT = './outputs/stage_c/asap2_report.json'

DA_HISTORY = './outputs_da/graphs/adaptation_history.json'
DA_EVAL = './outputs_da/graphs/evaluation_data_da.json'
DA_REPORT = './outputs_da/stage_c/asap2_da_report.json'

ASAP1_TSV = './data/training_set_rel3.tsv'

ASAP1_RATER_RANGES = {
    1: (1, 6), 2: (1, 6), 3: (0, 3), 4: (0, 3),
    5: (0, 4), 6: (0, 4), 7: (0, 12), 8: (0, 30),
}

ASAP2_PROMPT_NAMES = {
    100: 'Cowboy Waves',
    101: 'Car-free Cities',
    102: 'Electoral College',
    103: 'Driverless Cars',
    104: 'Exploring Venus',
    105: 'Facial Coding',
    106: 'Face on Mars',
}

# Style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

COLORS = {
    'zs': '#2196F3',       # blue
    'da': '#FF5722',       # orange
    'mse': '#4CAF50',      # green
    'ranking': '#9C27B0',  # purple
    'ordinal': '#FF9800',  # amber
    'soft_qwk': '#E91E63', # pink
    'dann': '#F44336',     # red
    'coral': '#00BCD4',    # cyan
    'total': '#333333',    # dark gray
}


def load_json(path):
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping related plots.")
        return None
    with open(path) as f:
        return json.load(f)


def save_fig(name):
    path = os.path.join(FIGURES_DIR, f'{name}.png')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")

def plot_zs_loss_curves(history):
    steps = history['step_logs']
    if not steps:
        return

    df = pd.DataFrame(steps)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Zero-Shot Training: Loss Components Over Steps', fontweight='bold')

    loss_keys = [
        ('loss_total', 'Total Loss', COLORS['total']),
        ('loss_mse', 'MSE Loss', COLORS['mse']),
        ('loss_ranking', 'Ranking Loss', COLORS['ranking']),
        ('loss_ordinal', 'Ordinal (CORN) Loss', COLORS['ordinal']),
        ('loss_soft_qwk', 'Soft QWK Loss', COLORS['soft_qwk']),
    ]

    for idx, (key, title, color) in enumerate(loss_keys):
        ax = axes[idx // 3, idx % 3]
        if key in df.columns:
            raw = df[key].values
            window = max(1, len(raw) // 50)
            smoothed = pd.Series(raw).rolling(window, min_periods=1).mean()
            ax.plot(df['global_step'], raw, alpha=0.2, color=color, linewidth=0.5)
            ax.plot(df['global_step'], smoothed, color=color, linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Global Step')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)

            if 'stage' in df.columns:
                for stage in df['stage'].unique():
                    first = df[df['stage'] == stage]['global_step'].iloc[0]
                    ax.axvline(x=first, color='gray', linestyle='--', alpha=0.5)
                    ax.text(first, ax.get_ylim()[1], f' {stage}', fontsize=8, va='top')

    ax = axes[1, 2]
    if 'lr' in df.columns:
        ax.plot(df['global_step'], df['lr'], color='#607D8B', linewidth=2)
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

    plt.tight_layout()
    save_fig('01_zs_loss_curves')

def plot_zs_qwk_epochs(history):
    epochs = history['epoch_logs']
    if not epochs:
        return

    df = pd.DataFrame(epochs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Zero-Shot Training: Validation Metrics Over Epochs', fontweight='bold')

    ax = axes[0]
    if 'val_qwk_ordinal' in df.columns:
        ax.plot(df['epoch'], df['val_qwk_ordinal'], 'o-', color=COLORS['ordinal'],
                label='Ordinal Head', linewidth=2, markersize=6)
    if 'val_qwk_regression' in df.columns:
        ax.plot(df['epoch'], df['val_qwk_regression'], 's-', color=COLORS['mse'],
                label='Regression Head', linewidth=2, markersize=6)
    if 'val_qwk_best' in df.columns:
        ax.plot(df['epoch'], df['val_qwk_best'], 'D-', color=COLORS['total'],
                label='Best', linewidth=2.5, markersize=7)
    ax.set_title('Validation QWK')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('QWK')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    ax = axes[1]
    if 'val_rmse' in df.columns:
        ax.plot(df['epoch'], df['val_rmse'], 'o-', color='#E91E63', linewidth=2, markersize=6)
    ax.set_title('Validation RMSE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (normalized)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    if 'train_total' in df.columns:
        ax.plot(df['epoch'], df['train_total'], 'o-', color=COLORS['zs'], label='Train', linewidth=2)
    if 'val_total' in df.columns:
        ax.plot(df['epoch'], df['val_total'], 's-', color=COLORS['da'], label='Val', linewidth=2)
    ax.set_title('Train vs Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('02_zs_qwk_epochs')

def plot_zs_per_prompt_epochs(history):
    epochs = history['epoch_logs']
    if not epochs:
        return

    df = pd.DataFrame(epochs)
    prompt_cols = [c for c in df.columns if c.startswith('val_qwk_prompt_')]

    if not prompt_cols:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.tab10

    for idx, col in enumerate(sorted(prompt_cols)):
        pid = int(col.split('_')[-1])
        label = f'Prompt {pid}'
        ax.plot(df['epoch'], df[col], 'o-', color=cmap(idx), label=label,
                linewidth=2, markersize=5)

    ax.set_title('Zero-Shot: Per-Prompt QWK on ASAP1 Val Over Epochs', fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('QWK')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig('03_zs_per_prompt_epochs')

def plot_da_dann_dynamics(history):
    steps = history['step_logs']
    epochs = history['epoch_logs']
    if not steps:
        return

    df_steps = pd.DataFrame(steps)
    df_epochs = pd.DataFrame(epochs) if epochs else None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Domain Adaptation: DANN + CORAL Training Dynamics', fontweight='bold')

    ax = axes[0, 0]
    if 'loss_dann' in df_steps.columns:
        raw = df_steps['loss_dann'].values
        window = max(1, len(raw) // 30)
        smoothed = pd.Series(raw).rolling(window, min_periods=1).mean()
        ax.plot(df_steps['global_step'], raw, alpha=0.15, color=COLORS['dann'], linewidth=0.5)
        ax.plot(df_steps['global_step'], smoothed, color=COLORS['dann'], linewidth=2)
        ax.axhline(y=np.log(2), color='black', linestyle='--', alpha=0.6, label=f'Random baseline (ln2={np.log(2):.3f})')
        ax.set_title('DANN Domain Loss')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Cross-Entropy')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if 'loss_coral' in df_steps.columns:
        raw = df_steps['loss_coral'].values
        window = max(1, len(raw) // 30)
        smoothed = pd.Series(raw).rolling(window, min_periods=1).mean()
        ax.plot(df_steps['global_step'], raw, alpha=0.15, color=COLORS['coral'], linewidth=0.5)
        ax.plot(df_steps['global_step'], smoothed, color=COLORS['coral'], linewidth=2)
        ax.set_title('CORAL Covariance Loss')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if 'dann_lambda' in df_steps.columns:
        ax.plot(df_steps['global_step'], df_steps['dann_lambda'],
                color='#9C27B0', linewidth=2)
        ax.set_title('GRL Lambda Schedule')
        ax.set_xlabel('Global Step')
        ax.set_ylabel('λ (gradient reversal strength)')
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for key, label, color in [
        ('loss_mse', 'MSE', COLORS['mse']),
        ('loss_soft_qwk', 'Soft QWK', COLORS['soft_qwk']),
        ('loss_ordinal', 'Ordinal', COLORS['ordinal']),
    ]:
        if key in df_steps.columns:
            raw = df_steps[key].values
            window = max(1, len(raw) // 30)
            smoothed = pd.Series(raw).rolling(window, min_periods=1).mean()
            ax.plot(df_steps['global_step'], smoothed, color=color, linewidth=2, label=label)
    ax.set_title('Scoring Losses During DA')
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if df_epochs is not None and 'val_qwk_best' in df_epochs.columns:
        ax.plot(df_epochs['epoch'], df_epochs['val_qwk_best'], 'D-',
                color=COLORS['da'], linewidth=2.5, markersize=8)
        ax.set_title('Val QWK During DA')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('QWK (ASAP1 val)')
        ax.grid(True, alpha=0.3)
        for i, row in df_epochs.iterrows():
            ax.annotate(f"{row['val_qwk_best']:.4f}",
                        (row['epoch'], row['val_qwk_best']),
                        textcoords="offset points", xytext=(0, 10),
                        ha='center', fontsize=9)

    ax = axes[1, 2]
    if df_epochs is not None and 'train_total' in df_epochs.columns:
        ax.plot(df_epochs['epoch'], df_epochs['train_total'], 'o-',
                color=COLORS['total'], linewidth=2, label='Total')
        if 'train_dann' in df_epochs.columns:
            ax.plot(df_epochs['epoch'], df_epochs['train_dann'], 's-',
                    color=COLORS['dann'], linewidth=2, label='DANN')
        if 'train_coral' in df_epochs.columns:
            ax.plot(df_epochs['epoch'], df_epochs['train_coral'], '^-',
                    color=COLORS['coral'], linewidth=2, label='CORAL')
        ax.set_title('Epoch-Level Losses (DA)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('04_da_dann_dynamics')

def plot_ust_progress(history):
    ust = history.get('ust_logs', [])
    if not ust:
        print("  No UST logs found, skipping.")
        return

    df = pd.DataFrame(ust)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('UST Self-Training Progress', fontweight='bold')

    ax = axes[0]
    x = df['iteration']
    ax.plot(x, df['qwk_before'], 'o--', color='gray', label='Before', linewidth=2)
    ax.plot(x, df['qwk_after'], 'D-', color=COLORS['da'], label='After', linewidth=2.5)
    for _, row in df.iterrows():
        delta = row['qwk_after'] - row['qwk_before']
        symbol = '↑' if delta > 0 else '↓'
        ax.annotate(f"{symbol}{abs(delta):.4f}",
                    (row['iteration'], row['qwk_after']),
                    textcoords="offset points", xytext=(10, 5), fontsize=9)
    ax.set_title('QWK Before/After Each UST Iteration')
    ax.set_xlabel('UST Iteration')
    ax.set_ylabel('Val QWK')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, df['n_pseudo_labels'], color=COLORS['coral'], alpha=0.8)
    ax.set_title('Pseudo-Labels Selected')
    ax.set_xlabel('UST Iteration')
    ax.set_ylabel('Number of Pseudo-Labeled ASAP2 Essays')
    for bar, n in zip(bars, df['n_pseudo_labels']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{int(n):,}', ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    ax.plot(x, df['mean_uncertainty'], 'o-', color='#9C27B0', linewidth=2, label='Mean')
    ax.plot(x, df['uncertainty_threshold'], 's--', color='#F44336', linewidth=2, label='Threshold')
    ax.set_title('MC Dropout Uncertainty')
    ax.set_xlabel('UST Iteration')
    ax.set_ylabel('Prediction Std Dev')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('05_ust_progress')

def plot_model_comparison_overall(zs_eval, da_eval):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle('Zero-Shot vs Domain-Adapted: ASAP2 Performance', fontweight='bold')

    ax = axes[0]
    methods = ['ordinal', 'simple_round', 'threshold']
    labels = ['Ordinal Head', 'Regression\n(Rounded)', 'Regression\n(Threshold)']

    zs_vals = [zs_eval.get(f'asap2_qwk_{m}', 0) for m in methods]
    da_vals = [da_eval.get(f'asap2_qwk_{m}', 0) for m in methods]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, zs_vals, w, label='Zero-Shot', color=COLORS['zs'], alpha=0.85)
    bars2 = ax.bar(x + w/2, da_vals, w, label='Domain-Adapted', color=COLORS['da'], alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title('QWK by Scoring Method')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Quadratic Weighted Kappa')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(max(zs_vals), max(da_vals)) * 1.2)

    ax = axes[1]
    zs_best = max(zs_vals)
    da_best = max(da_vals)
    bars = ax.bar(['Zero-Shot\n(Best)', 'Domain-Adapted\n(Best)'],
                  [zs_best, da_best],
                  color=[COLORS['zs'], COLORS['da']], alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    delta = da_best - zs_best
    sign = '+' if delta >= 0 else ''
    ax.set_title(f'Best QWK Comparison (Δ = {sign}{delta:.4f})')
    ax.set_ylabel('QWK')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(zs_best, da_best) * 1.3)

    plt.tight_layout()
    save_fig('06_model_comparison_overall')

def plot_per_prompt_comparison(zs_eval, da_eval):
    prompt_ids = sorted([k for k in zs_eval.keys() if k.startswith('asap2_qwk_prompt_')])
    if not prompt_ids:
        return

    pids = [int(k.split('_')[-1]) for k in prompt_ids]
    labels = [ASAP2_PROMPT_NAMES.get(p, f'P{p}') for p in pids]
    zs_qwk = [zs_eval.get(f'asap2_qwk_prompt_{p}', 0) for p in pids]
    da_qwk = [da_eval.get(f'asap2_qwk_prompt_{p}', 0) for p in pids]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(pids))
    w = 0.35
    bars1 = ax.bar(x - w/2, zs_qwk, w, label='Zero-Shot', color=COLORS['zs'], alpha=0.85)
    bars2 = ax.bar(x + w/2, da_qwk, w, label='Domain-Adapted', color=COLORS['da'], alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    true_scores = np.array(zs_eval.get('asap2_true_scores', []))
    prompt_ids_arr = np.array(zs_eval.get('asap2_prompt_ids', []))
    for i, pid in enumerate(pids):
        n = (prompt_ids_arr == pid).sum()
        ax.text(i, -0.03, f'n={n}', ha='center', fontsize=8, color='gray')

    ax.set_title('Per-Prompt QWK on ASAP2: Zero-Shot vs Domain-Adapted', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('QWK')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()
    save_fig('07_per_prompt_comparison')

def plot_confusion_matrices(zs_eval, da_eval):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrices: True vs Predicted ASAP2 Scores', fontweight='bold')

    for idx, (eval_data, title, ax) in enumerate([
        (zs_eval, 'Zero-Shot', axes[0]),
        (da_eval, 'Domain-Adapted', axes[1]),
    ]):
        true = np.array(eval_data['asap2_true_scores'])
        pred = np.array(eval_data['asap2_pred_best'])
        labels = sorted(set(true) | set(pred))

        cm = confusion_matrix(true, pred, labels=labels)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = cm[i, j]
                pct = cm_norm[i, j]
                color = 'white' if pct > 0.5 else 'black'
                ax.text(j, i, f'{val}', ha='center', va='center',
                        fontsize=7, color=color)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted Score')
        ax.set_ylabel('True Score')

        qwk = cohen_kappa_score(true, pred, weights='quadratic')
        ax.set_title(f'{title}\nQWK = {qwk:.4f}')

    fig.colorbar(im, ax=axes, label='Row-Normalized Frequency', shrink=0.8)
    plt.tight_layout()
    save_fig('08_confusion_matrices')

def plot_score_distributions(zs_eval, da_eval):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Score Distributions: True vs Predicted', fontweight='bold')

    true = np.array(zs_eval['asap2_true_scores'])
    bins = np.arange(0.5, 7.5, 1)

    for idx, (eval_data, model_name, row) in enumerate([
        (zs_eval, 'Zero-Shot', 0),
        (da_eval, 'Domain-Adapted', 1),
    ]):
        pred = np.array(eval_data['asap2_pred_best'])

        ax = axes[row, 0]
        ax.hist(true, bins=bins, alpha=0.6, color='steelblue', label='True', edgecolor='white')
        ax.hist(pred, bins=bins, alpha=0.6, color=COLORS['da'] if row else COLORS['zs'],
                label='Predicted', edgecolor='white')
        ax.set_title(f'{model_name}: Score Distribution')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.grid(True, alpha=0.3, axis='y')

        ax = axes[row, 1]
        jitter_t = true + np.random.normal(0, 0.12, len(true))
        jitter_p = pred + np.random.normal(0, 0.12, len(pred))
        ax.scatter(jitter_t, jitter_p, alpha=0.02, s=3, c='steelblue')
        ax.plot([0.5, 6.5], [0.5, 6.5], 'r--', linewidth=2, label='Perfect')

        for s in [1, 2, 3, 4, 5, 6]:
            mask = true == s
            if mask.sum() > 0:
                mean_pred = pred[mask].mean()
                ax.plot(s, mean_pred, 'D', color='red', markersize=8, zorder=5)

        ax.set_title(f'{model_name}: True vs Predicted')
        ax.set_xlabel('True Score')
        ax.set_ylabel('Predicted Score')
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0.5, 6.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('09_score_distributions')

def plot_error_analysis(zs_eval, da_eval):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Error Analysis: Zero-Shot vs Domain-Adapted', fontweight='bold')

    ax = axes[0, 0]
    categories = ['Exact', 'Off by 1', 'Off by 2', 'Off by 3', 'Off by 4+']
    for idx, (eval_data, name, color) in enumerate([
        (zs_eval, 'Zero-Shot', COLORS['zs']),
        (da_eval, 'Domain-Adapted', COLORS['da']),
    ]):
        true = np.array(eval_data['asap2_true_scores'])
        pred = np.array(eval_data['asap2_pred_best'])
        abs_err = np.abs(true - pred)

        counts = [
            np.mean(abs_err == 0),
            np.mean(abs_err == 1),
            np.mean(abs_err == 2),
            np.mean(abs_err == 3),
            np.mean(abs_err >= 4),
        ]

        x = np.arange(len(categories))
        w = 0.35
        offset = -w/2 + idx * w
        bars = ax.bar(x + offset, [c * 100 for c in counts], w,
                       label=name, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 1:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                        f'{h:.1f}%', ha='center', fontsize=8)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(categories)
    ax.set_ylabel('Percentage of Predictions')
    ax.set_title('Error Magnitude Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[0, 1]
    for eval_data, name, color, marker in [
        (zs_eval, 'Zero-Shot', COLORS['zs'], 'o'),
        (da_eval, 'Domain-Adapted', COLORS['da'], 's'),
    ]:
        true = np.array(eval_data['asap2_true_scores'])
        pred = np.array(eval_data['asap2_pred_best'])
        bias_by_score = []
        for s in [1, 2, 3, 4, 5, 6]:
            mask = true == s
            if mask.sum() > 0:
                bias_by_score.append(pred[mask].mean() - s)
            else:
                bias_by_score.append(0)
        ax.plot([1, 2, 3, 4, 5, 6], bias_by_score, f'{marker}-',
                color=color, linewidth=2, markersize=8, label=name)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_title('Prediction Bias by True Score')
    ax.set_xlabel('True Score')
    ax.set_ylabel('Mean Bias (Predicted − True)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for eval_data, name, color, marker in [
        (zs_eval, 'Zero-Shot', COLORS['zs'], 'o'),
        (da_eval, 'Domain-Adapted', COLORS['da'], 's'),
    ]:
        true = np.array(eval_data['asap2_true_scores'])
        pred = np.array(eval_data['asap2_pred_best'])
        mae_by_score = []
        for s in [1, 2, 3, 4, 5, 6]:
            mask = true == s
            if mask.sum() > 0:
                mae_by_score.append(np.abs(pred[mask] - s).mean())
            else:
                mae_by_score.append(0)
        ax.plot([1, 2, 3, 4, 5, 6], mae_by_score, f'{marker}-',
                color=color, linewidth=2, markersize=8, label=name)

    ax.set_title('MAE by True Score')
    ax.set_xlabel('True Score')
    ax.set_ylabel('Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for eval_data, name, color in [
        (zs_eval, 'Zero-Shot', COLORS['zs']),
        (da_eval, 'Domain-Adapted', COLORS['da']),
    ]:
        true = np.array(eval_data['asap2_true_scores'])
        pred = np.array(eval_data['asap2_pred_best'])
        residuals = pred - true
        ax.hist(residuals, bins=np.arange(-5.5, 6.5, 1), alpha=0.5,
                color=color, label=f'{name} (μ={residuals.mean():.2f})', edgecolor='white')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_title('Residual Distribution (Predicted − True)')
    ax.set_xlabel('Residual')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig('10_error_analysis')

def plot_calibration(zs_eval, da_eval):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Calibration: Mean Predicted Score vs True Score', fontweight='bold')

    for idx, (eval_data, name, color) in enumerate([
        (zs_eval, 'Zero-Shot', COLORS['zs']),
        (da_eval, 'Domain-Adapted', COLORS['da']),
    ]):
        ax = axes[idx]
        true = np.array(eval_data['asap2_true_scores'])
        pred_reg = np.array(eval_data['asap2_pred_reg'])

        for s in [1, 2, 3, 4, 5, 6]:
            mask = true == s
            if mask.sum() > 0:
                vals = pred_reg[mask]
                bp = ax.boxplot([vals], positions=[s], widths=0.6,
                                patch_artist=True, showfliers=False)
                bp['boxes'][0].set_facecolor(color)
                bp['boxes'][0].set_alpha(0.4)
                bp['medians'][0].set_color('black')

        ax.plot([0.5, 6.5], [0.5, 6.5], 'r--', linewidth=2, label='Perfect calibration')
        ax.set_title(f'{name}')
        ax.set_xlabel('True Score')
        ax.set_ylabel('Predicted Score (Regression)')
        ax.set_xlim(0.5, 6.5)
        ax.set_ylim(0.5, 6.5)
        ax.set_aspect('equal')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('11_calibration')

def plot_asap1_analysis():
    if not os.path.exists(ASAP1_TSV):
        print(f"  WARNING: {ASAP1_TSV} not found, skipping ASAP1 plots.")
        return

    df = pd.read_csv(ASAP1_TSV, sep='\t', encoding='ISO-8859-1')
    df['avg_score'] = (df['rater1_domain1'] + df['rater2_domain1']) / 2.0

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('ASAP1 Dataset: Averaged Rater Score Distributions by Prompt', fontweight='bold')

    for idx, pid in enumerate(sorted(df['essay_set'].unique())):
        ax = axes[idx // 4, idx % 4]
        sub = df[df['essay_set'] == pid]
        lo, hi = ASAP1_RATER_RANGES[pid]

        bins = np.arange(lo - 0.25, hi + 0.75, 0.5)
        ax.hist(sub['avg_score'], bins=bins, color=plt.cm.Set2(idx),
                edgecolor='white', alpha=0.85)
        ax.set_title(f'Prompt {pid} (n={len(sub)})')
        ax.set_xlabel(f'Avg Score [{lo}–{hi}]')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')

        mean_val = sub['avg_score'].mean()
        ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=1.5)
        ax.text(mean_val, ax.get_ylim()[1] * 0.9, f'μ={mean_val:.1f}',
                ha='center', fontsize=8, color='red')

    plt.tight_layout()
    save_fig('12_asap1_score_distributions')

def plot_rater_agreement():
    if not os.path.exists(ASAP1_TSV):
        return

    df = pd.read_csv(ASAP1_TSV, sep='\t', encoding='ISO-8859-1')

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('ASAP1: Rater 1 vs Rater 2 Agreement by Prompt', fontweight='bold')

    for idx, pid in enumerate(sorted(df['essay_set'].unique())):
        ax = axes[idx // 4, idx % 4]
        sub = df[df['essay_set'] == pid]
        r1 = sub['rater1_domain1'].values
        r2 = sub['rater2_domain1'].values

        jitter1 = r1 + np.random.normal(0, 0.1, len(r1))
        jitter2 = r2 + np.random.normal(0, 0.1, len(r2))
        ax.scatter(jitter1, jitter2, alpha=0.15, s=5, c=plt.cm.Set2(idx))

        lo, hi = ASAP1_RATER_RANGES[pid]
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5)

        qwk = cohen_kappa_score(r1, r2, weights='quadratic')
        exact = np.mean(r1 == r2)
        ax.set_title(f'P{pid}: QWK={qwk:.3f}, exact={exact:.1%}')
        ax.set_xlabel('Rater 1')
        ax.set_ylabel('Rater 2')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('13_rater_agreement')

def plot_cross_dataset_distributions(zs_eval):
    if not os.path.exists(ASAP1_TSV):
        return

    df = pd.read_csv(ASAP1_TSV, sep='\t', encoding='ISO-8859-1')
    df['avg_score'] = (df['rater1_domain1'] + df['rater2_domain1']) / 2.0

    all_norms = []
    for _, row in df.iterrows():
        lo, hi = ASAP1_RATER_RANGES[row['essay_set']]
        all_norms.append((row['avg_score'] - lo) / (hi - lo))
    asap1_norms = np.array(all_norms)

    true_asap2 = np.array(zs_eval['asap2_true_scores'])
    asap2_norms = (true_asap2 - 1) / 5.0  # ASAP2: 1-6 → 0-1

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('ASAP1 vs ASAP2: Score Distribution Comparison', fontweight='bold')

    ax = axes[0]
    bins = np.linspace(0, 1, 25)
    ax.hist(asap1_norms, bins=bins, alpha=0.6, color=COLORS['zs'],
            label=f'ASAP1 (n={len(asap1_norms):,})', density=True, edgecolor='white')
    ax.hist(asap2_norms, bins=bins, alpha=0.6, color=COLORS['da'],
            label=f'ASAP2 (n={len(asap2_norms):,})', density=True, edgecolor='white')
    ax.set_title('Normalized Score Distributions')
    ax.set_xlabel('Normalized Score [0, 1]')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.hist(true_asap2, bins=np.arange(0.5, 7.5, 1), color=COLORS['da'],
            edgecolor='white', alpha=0.85)
    ax.set_title('ASAP2 Score Distribution (Native 1-6)')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[2]
    asap1_lens = df['essay'].str.split().str.len().values
    ax.hist(asap1_lens, bins=50, alpha=0.7, color=COLORS['zs'], label='ASAP1', density=True)
    ax.set_title('ASAP1 Essay Length Distribution')
    ax.set_xlabel('Word Count')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig('14_cross_dataset_distributions')

def plot_quantile_bins(zs_history):
    metadata = zs_history.get('metadata', {})
    bin_edges = metadata.get('bin_edges', None)

    if not bin_edges:
        return

    if not os.path.exists(ASAP1_TSV):
        return

    df = pd.read_csv(ASAP1_TSV, sep='\t', encoding='ISO-8859-1')
    df['avg_score'] = (df['rater1_domain1'] + df['rater2_domain1']) / 2.0

    all_norms = []
    for _, row in df.iterrows():
        lo, hi = ASAP1_RATER_RANGES[row['essay_set']]
        all_norms.append((row['avg_score'] - lo) / (hi - lo))
    norms = np.array(all_norms)
    labels = np.digitize(norms, bin_edges)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Quantile Binning for Ordinal Regression', fontweight='bold')

    ax = axes[0]
    ax.hist(norms, bins=60, color='steelblue', alpha=0.7, edgecolor='white')
    for i, edge in enumerate(bin_edges):
        ax.axvline(x=edge, color='red', linestyle='--', linewidth=2)
        ax.text(edge, ax.get_ylim()[1] * 0.95, f'  {edge:.3f}',
                fontsize=8, color='red', rotation=90, va='top')
    ax.set_title('Normalized Score Distribution with Bin Edges')
    ax.set_xlabel('Normalized Score [0, 1]')
    ax.set_ylabel('Count')
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    K = len(bin_edges) + 1
    counts = [(labels == k).sum() for k in range(K)]
    pcts = [100 * c / len(labels) for c in counts]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, K))
    bars = ax.bar(range(K), pcts, color=colors, edgecolor='white')
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{pct:.1f}%\n({cnt:,})', ha='center', fontsize=9)
    ax.set_title(f'Bin Population ({K} bins)')
    ax.set_xlabel('Ordinal Bin')
    ax.set_ylabel('Percentage')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=100/K, color='red', linestyle='--', alpha=0.5,
               label=f'Uniform ({100/K:.1f}%)')
    ax.legend()

    plt.tight_layout()
    save_fig('15_quantile_bins')

def plot_regression_outputs(zs_eval, da_eval):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Continuous Regression Output Distribution on ASAP2', fontweight='bold')

    for idx, (eval_data, name, color) in enumerate([
        (zs_eval, 'Zero-Shot', COLORS['zs']),
        (da_eval, 'Domain-Adapted', COLORS['da']),
    ]):
        ax = axes[idx]
        pred_reg = np.array(eval_data['asap2_pred_reg'])
        true = np.array(eval_data['asap2_true_scores'])

        ax.hist(pred_reg, bins=50, alpha=0.7, color=color, edgecolor='white', label='Predicted')

        for s in [1, 2, 3, 4, 5, 6]:
            ax.axvline(x=s, color='gray', linestyle=':', alpha=0.5)

        ax.set_title(f'{name}: Regression Output (μ={pred_reg.mean():.2f}, σ={pred_reg.std():.2f})')
        ax.set_xlabel('Predicted Score (continuous)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig('16_regression_outputs')

def plot_per_prompt_details(zs_eval, da_eval):
    true = np.array(zs_eval['asap2_true_scores'])
    pids = np.array(zs_eval['asap2_prompt_ids'])
    unique_pids = sorted(np.unique(pids))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Per-Prompt Score Distributions: True vs Zero-Shot vs DA Predicted', fontweight='bold')

    for idx, pid in enumerate(unique_pids):
        if idx >= 7:
            break
        ax = axes[idx // 4, idx % 4]
        mask = pids == pid
        t = true[mask]
        zs_p = np.array(zs_eval['asap2_pred_best'])[mask]
        da_p = np.array(da_eval['asap2_pred_best'])[mask]

        bins = np.arange(0.5, 7.5, 1)
        ax.hist(t, bins=bins, alpha=0.5, color='gray', label='True', edgecolor='white')
        ax.hist(zs_p, bins=bins, alpha=0.4, color=COLORS['zs'], label='ZS', edgecolor='white')
        ax.hist(da_p, bins=bins, alpha=0.4, color=COLORS['da'], label='DA', edgecolor='white')

        name = ASAP2_PROMPT_NAMES.get(pid, f'P{pid}')
        ax.set_title(f'{name} (n={mask.sum()})', fontsize=10)
        ax.set_xlabel('Score')
        ax.set_xticks([1, 2, 3, 4, 5, 6])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    if len(unique_pids) < 8:
        axes[1, 3].axis('off')

    plt.tight_layout()
    save_fig('17_per_prompt_details')

def plot_combined_qwk_timeline(zs_history, da_history):
    zs_epochs = pd.DataFrame(zs_history.get('epoch_logs', []))
    da_epochs = pd.DataFrame(da_history.get('epoch_logs', []))

    if zs_epochs.empty or da_epochs.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    zs_x = zs_epochs['epoch'].values
    if 'val_qwk_best' in zs_epochs.columns:
        ax.plot(zs_x, zs_epochs['val_qwk_best'], 'o-', color=COLORS['zs'],
                linewidth=2.5, markersize=8, label='Stage S (Zero-Shot)', zorder=3)

    offset = zs_x[-1] if len(zs_x) > 0 else 0
    da_x = da_epochs['epoch'].values + offset
    if 'val_qwk_best' in da_epochs.columns:
        ax.plot(da_x, da_epochs['val_qwk_best'], 's-', color=COLORS['da'],
                linewidth=2.5, markersize=8, label='Stage U (DANN+CORAL)', zorder=3)

    ust_logs = da_history.get('ust_logs', [])
    if ust_logs:
        ust_x_start = da_x[-1] if len(da_x) > 0 else offset
        for i, ust in enumerate(ust_logs):
            ust_x = ust_x_start + i + 1
            ax.plot(ust_x, ust['qwk_after'], 'D', color='#9C27B0',
                    markersize=10, zorder=4)
            if i == 0:
                ax.plot(ust_x, ust['qwk_after'], 'D', color='#9C27B0',
                        markersize=10, label='UST Iterations')

    ax.axvline(x=offset + 0.5, color='gray', linestyle='--', alpha=0.5)
    ax.text(offset * 0.5, ax.get_ylim()[0], 'Stage S\n(Supervised)',
            ha='center', fontsize=10, color=COLORS['zs'], alpha=0.7)
    ax.text(offset + (da_x[-1] - offset) / 2 if len(da_x) > 0 else offset + 1,
            ax.get_ylim()[0], 'Stage U\n(DA)',
            ha='center', fontsize=10, color=COLORS['da'], alpha=0.7)

    ax.set_title('Full Training Timeline: Val QWK Progression', fontweight='bold')
    ax.set_xlabel('Epoch (Combined)')
    ax.set_ylabel('Val QWK (ASAP1)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig('18_combined_qwk_timeline')

def plot_summary_dashboard(zs_eval, da_eval, zs_history, da_history):
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('ASAP1 → ASAP2 Transfer: Summary Dashboard', fontweight='bold', fontsize=15)

    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0:2])
    zs_best = max(zs_eval.get('asap2_qwk_ordinal', 0),
                  zs_eval.get('asap2_qwk_simple', 0),
                  zs_eval.get('asap2_qwk_threshold', 0))
    da_best = max(da_eval.get('asap2_qwk_ordinal', 0),
                  da_eval.get('asap2_qwk_simple', 0),
                  da_eval.get('asap2_qwk_threshold', 0))

    bars = ax.bar(['Zero-Shot', 'Domain-Adapted'], [zs_best, da_best],
                  color=[COLORS['zs'], COLORS['da']], alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_title('ASAP2 Best QWK', fontweight='bold')
    ax.set_ylabel('QWK')
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[0, 2:4])
    zs_val = zs_eval.get('asap1_val_qwk_ordinal',
                          zs_eval.get('asap1_val_qwk', 0))
    da_val = da_eval.get('asap1_val_qwk', 0)
    bars = ax.bar(['ZS: ASAP1 Val', 'DA: ASAP1 Val'], [zs_val, da_val],
                  color=[COLORS['zs'], COLORS['da']], alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h:.4f}', ha='center', fontsize=12, fontweight='bold')
    ax.set_title('ASAP1 Validation QWK', fontweight='bold')
    ax.set_ylabel('QWK')
    ax.grid(True, alpha=0.3, axis='y')

    ax = fig.add_subplot(gs[1, :])
    prompt_keys = sorted([k for k in zs_eval.keys() if k.startswith('asap2_qwk_prompt_')])
    if prompt_keys:
        pids = [int(k.split('_')[-1]) for k in prompt_keys]
        labels = [ASAP2_PROMPT_NAMES.get(p, f'P{p}') for p in pids]
        zs_qwk = [zs_eval.get(f'asap2_qwk_prompt_{p}', 0) for p in pids]
        da_qwk = [da_eval.get(f'asap2_qwk_prompt_{p}', 0) for p in pids]

        x = np.arange(len(pids))
        w = 0.35
        ax.bar(x - w/2, zs_qwk, w, label='Zero-Shot', color=COLORS['zs'], alpha=0.85)
        ax.bar(x + w/2, da_qwk, w, label='Domain-Adapted', color=COLORS['da'], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylabel('QWK')
        ax.set_title('Per-Prompt QWK Comparison', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)

    ax = fig.add_subplot(gs[2, 0:2])
    ax.axis('off')
    table_data = []
    for eval_data, name in [(zs_eval, 'Zero-Shot'), (da_eval, 'DA')]:
        true = np.array(eval_data['asap2_true_scores'])
        pred = np.array(eval_data['asap2_pred_best'])
        abs_err = np.abs(true - pred)
        bias = (pred - true).mean()
        table_data.append([
            name,
            f"{max(eval_data.get('asap2_qwk_ordinal', 0), eval_data.get('asap2_qwk_simple', 0), eval_data.get('asap2_qwk_threshold', 0)):.4f}",
            f"{np.mean(abs_err == 0):.1%}",
            f"{np.mean(abs_err <= 1):.1%}",
            f"{abs_err.mean():.3f}",
            f"{bias:+.3f}",
        ])
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Best QWK', 'Exact', 'Adjacent', 'MAE', 'Bias'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    ax.set_title('Summary Metrics', fontweight='bold', pad=20)
    ax = fig.add_subplot(gs[2, 2:4])
    ax.axis('off')
    n_asap1 = zs_history.get('metadata', {}).get('n_train', '?')
    n_asap2 = len(zs_eval.get('asap2_true_scores', []))
    bin_edges = zs_history.get('metadata', {}).get('bin_edges', [])

    n_asap1_str = f"{n_asap1:,}" if isinstance(n_asap1, int) else str(n_asap1)
    info_text = (
        f"Dataset Summary\n"
        f"{'─' * 40}\n"
        f"ASAP1 training essays:   {n_asap1_str}\n"
        f"ASAP2 evaluation essays: {n_asap2:,}\n"
        f"Ordinal bins:            {len(bin_edges) + 1}\n"
        f"Bin edges:               {[round(e, 3) for e in bin_edges]}\n"
        f"\n"
        f"Architecture\n"
        f"{'─' * 40}\n"
        f"Backbone: Longformer-base-4096\n"
        f"Heads: Regression + CORN Ordinal\n"
        f"LoRA rank: 8\n"
        f"Losses: MSE + Ranking + CORN + Soft QWK\n"
        f"DA: DANN + Deep CORAL + UST"
    )
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    save_fig('19_summary_dashboard')

def plot_ordinal_bin_predictions(zs_eval, da_eval):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ordinal Bin Assignment: True vs Predicted', fontweight='bold')

    true = np.array(zs_eval['asap2_true_scores'])
    true_ord = true - 1  # 1-6 → 0-5

    for idx, (eval_data, name, color) in enumerate([
        (zs_eval, 'Zero-Shot', COLORS['zs']),
        (da_eval, 'Domain-Adapted', COLORS['da']),
    ]):
        ax = axes[idx]
        pred_ord = np.array(eval_data.get('asap2_pred_ordinal', eval_data['asap2_pred_best'])) - 1

        K = 6
        true_dist = [np.mean(true_ord == k) for k in range(K)]
        pred_dist = [np.mean(pred_ord == k) for k in range(K)]

        x = np.arange(K)
        w = 0.35
        ax.bar(x - w/2, [d * 100 for d in true_dist], w, label='True', color='gray', alpha=0.7)
        ax.bar(x + w/2, [d * 100 for d in pred_dist], w, label='Predicted', color=color, alpha=0.85)
        ax.set_title(f'{name}')
        ax.set_xlabel('Score (1-6)')
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in range(K)])
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_fig('20_ordinal_bin_predictions')

def main():
    print("=" * 60)
    print("Generating comprehensive visualizations...")
    print(f"Saving to {FIGURES_DIR}/")
    print("=" * 60)

    zs_history = load_json(ZS_HISTORY)
    zs_eval = load_json(ZS_EVAL)
    zs_report = load_json(ZS_REPORT)
    da_history = load_json(DA_HISTORY)
    da_eval = load_json(DA_EVAL)
    da_report = load_json(DA_REPORT)

    print()

    if zs_history:
        print("[1/10] Zero-shot training loss curves...")
        plot_zs_loss_curves(zs_history)

        print("[2/10] Zero-shot QWK over epochs...")
        plot_zs_qwk_epochs(zs_history)

        print("[3/10] Zero-shot per-prompt QWK over epochs...")
        plot_zs_per_prompt_epochs(zs_history)

    if da_history:
        print("[4/10] DA DANN/CORAL dynamics...")
        plot_da_dann_dynamics(da_history)

        print("[5/10] UST self-training progress...")
        plot_ust_progress(da_history)

    if zs_eval and da_eval:
        print("[6/10] Model comparison...")
        plot_model_comparison_overall(zs_eval, da_eval)
        plot_per_prompt_comparison(zs_eval, da_eval)
        plot_confusion_matrices(zs_eval, da_eval)
        plot_score_distributions(zs_eval, da_eval)
        plot_error_analysis(zs_eval, da_eval)
        plot_calibration(zs_eval, da_eval)
        plot_per_prompt_details(zs_eval, da_eval)
        plot_regression_outputs(zs_eval, da_eval)
        plot_ordinal_bin_predictions(zs_eval, da_eval)

    print("[7/10] ASAP1 dataset analysis...")
    plot_asap1_analysis()
    plot_rater_agreement()

    if zs_eval:
        print("[8/10] Cross-dataset distributions...")
        plot_cross_dataset_distributions(zs_eval)

    if zs_history:
        print("[9/10] Quantile bin visualization...")
        plot_quantile_bins(zs_history)

    if zs_history and da_history:
        print("[10/10] Combined timeline and summary dashboard...")
        plot_combined_qwk_timeline(zs_history, da_history)

    if zs_eval and da_eval and zs_history:
        plot_summary_dashboard(zs_eval, da_eval, zs_history,
                               da_history or {'epoch_logs': [], 'ust_logs': [], 'metadata': {}})

    n_figures = len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')])
    print(f"\n{'=' * 60}")
    print(f"Done! Generated {n_figures} figures in {FIGURES_DIR}/")
    print(f"{'=' * 60}")

    for f in sorted(os.listdir(FIGURES_DIR)):
        if f.endswith('.png'):
            size_kb = os.path.getsize(os.path.join(FIGURES_DIR, f)) / 1024
            print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
