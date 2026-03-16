import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import os
import json
import time
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from typing import Dict, List, Tuple

from config import TrainConfig, ModelConfig, Paths, get_device
from losses import AESMultiTaskLoss, SoftQWKLoss, corn_loss, PairwiseRankingLoss
from data import (
    build_dataset, load_asap1_labeled, split_asap1_train_val,
    load_asap2
)

class CORALLoss(nn.Module):

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        d = source_features.size(1)
        cs = self._covariance(source_features)
        ct = self._covariance(target_features)
        diff = cs - ct
        return (diff * diff).sum() / (4 * d * d)

    @staticmethod
    def _covariance(x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        if n < 2:
            return torch.zeros(x.size(1), x.size(1), device=x.device)
        x_centered = x - x.mean(dim=0, keepdim=True)
        return (x_centered.T @ x_centered) / (n - 1)

def dann_lambda_schedule(progress: float, gamma: float = 10.0) -> float:
    return 2.0 / (1.0 + np.exp(-gamma * progress)) - 1.0

class AdaptationHistory:
    def __init__(self):
        self.step_logs: List[dict] = []
        self.epoch_logs: List[dict] = []
        self.ust_logs: List[dict] = []
        self.metadata: dict = {}

    def log_step(self, step, epoch, losses, lr, dann_lambda):
        entry = {
            'global_step': step,
            'epoch': epoch,
            'lr': lr,
            'dann_lambda': dann_lambda,
        }
        for k, v in losses.items():
            entry[f'loss_{k}'] = float(v) if not isinstance(v, float) else v
        self.step_logs.append(entry)

    def log_epoch(self, epoch, train_metrics, val_metrics):
        entry = {'epoch': epoch}
        for k, v in train_metrics.items():
            entry[f'train_{k}'] = float(v)
        for k, v in val_metrics.items():
            entry[f'val_{k}'] = float(v)
        self.epoch_logs.append(entry)

    def log_ust_iteration(self, iteration, n_pseudo, mean_uncertainty,
                          threshold, qwk_before, qwk_after):
        self.ust_logs.append({
            'iteration': iteration,
            'n_pseudo_labels': n_pseudo,
            'mean_uncertainty': float(mean_uncertainty),
            'uncertainty_threshold': float(threshold),
            'qwk_before': float(qwk_before),
            'qwk_after': float(qwk_after),
        })

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'metadata': self.metadata,
                'step_logs': self.step_logs,
                'epoch_logs': self.epoch_logs,
                'ust_logs': self.ust_logs,
            }, f, indent=2)
        print(f"  Adaptation history saved to {path}")

def train_dann_epoch(
    model,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    model_cfg: ModelConfig,
    epoch: int,
    total_epochs: int,
    history: AdaptationHistory,
    global_step: int,
    coral_loss_fn: CORALLoss,
) -> Tuple[Dict[str, float], int]:
    model.train()
    epoch_losses = {}
    total_steps = 0

    mse_loss_fn = nn.MSELoss()
    ranking_loss_fn = PairwiseRankingLoss(margin=cfg.ranking_margin)
    soft_qwk_loss_fn = SoftQWKLoss(num_classes=model_cfg.num_classes).to(device)
    domain_ce = nn.CrossEntropyLoss()

    target_iter = iter(target_loader)
    optimizer.zero_grad()

    pbar = tqdm(source_loader, desc=f"  DANN Epoch {epoch}", leave=False)

    for step, src_batch in enumerate(pbar):
        try:
            tgt_batch = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_batch = next(target_iter)

        p = ((epoch - 1) + step / len(source_loader)) / total_epochs
        dann_lam = dann_lambda_schedule(p) if cfg.dann_lambda_schedule else 1.0
        model.set_dann_lambda(dann_lam)

        src_ids = src_batch['input_ids'].to(device)
        src_mask = src_batch['attention_mask'].to(device)
        src_global = src_batch['global_attention_mask'].to(device)
        src_norm = src_batch['norm_score'].to(device)
        src_ord = src_batch['ordinal_label'].to(device)

        tgt_ids = tgt_batch['input_ids'].to(device)
        tgt_mask = tgt_batch['attention_mask'].to(device)
        tgt_global = tgt_batch['global_attention_mask'].to(device)

        with autocast(dtype=torch.float16, enabled=cfg.fp16):
            src_out = model(src_ids, src_mask, src_global)
            tgt_out = model(tgt_ids, tgt_mask, tgt_global)

            losses = {}

            losses['mse'] = mse_loss_fn(src_out['reg_score'], src_norm) * cfg.mse_weight

            if src_norm.size(0) >= 2:
                losses['ranking'] = ranking_loss_fn(src_out['reg_score'], src_norm) * cfg.ranking_weight
            else:
                losses['ranking'] = torch.tensor(0.0, device=device)

            losses['ordinal'] = corn_loss(
                src_out['ordinal_logits'], src_ord, model_cfg.num_classes
            ) * cfg.ordinal_weight

            losses['soft_qwk'] = soft_qwk_loss_fn(
                src_out['ordinal_logits'], src_ord
            ) * cfg.soft_qwk_weight

            src_domain = torch.zeros(src_ids.size(0), dtype=torch.long, device=device)
            tgt_domain = torch.ones(tgt_ids.size(0), dtype=torch.long, device=device)
            all_domain_logits = torch.cat([src_out['domain_logits'], tgt_out['domain_logits']])
            all_domain_labels = torch.cat([src_domain, tgt_domain])
            losses['dann'] = domain_ce(all_domain_logits, all_domain_labels) * cfg.dann_weight

            if src_out['features'].size(0) > 1 and tgt_out['features'].size(0) > 1:
                losses['coral'] = coral_loss_fn(
                    src_out['features'], tgt_out['features']
                ) * cfg.coral_weight
            else:
                losses['coral'] = torch.tensor(0.0, device=device)

            losses['total'] = sum(losses.values())
            loss = losses['total'] / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            global_step += 1

        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
        total_steps += 1

        if total_steps % cfg.log_every_n_steps == 0:
            lr = optimizer.param_groups[0]['lr']
            history.log_step(
                global_step, epoch,
                {k: v.item() for k, v in losses.items()},
                lr, dann_lam,
            )

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'dann': f"{losses['dann'].item():.4f}",
            'coral': f"{losses['coral'].item():.4f}",
            'λ': f"{dann_lam:.3f}",
        })

    avg = {k: v / max(total_steps, 1) for k, v in epoch_losses.items()}
    return avg, global_step

@torch.no_grad()
def evaluate_da(model, loader, device, cfg, num_classes=6) -> Dict[str, float]:
    model.eval()
    all_preds_reg = []
    all_preds_ord = []
    all_true_norm = []
    all_true_ord = []
    all_prompt_ids = []

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)

        with autocast(dtype=torch.float16, enabled=cfg.fp16):
            preds = model.predict(input_ids, attention_mask, global_attention_mask)

        all_preds_reg.append(preds['reg_score'].cpu())
        all_preds_ord.append(preds['ordinal_labels'].cpu())
        all_true_norm.append(batch['norm_score'])
        all_true_ord.append(batch['ordinal_label'])
        all_prompt_ids.append(batch['prompt_id'])

    all_preds_reg = torch.cat(all_preds_reg).numpy()
    all_preds_ord = torch.cat(all_preds_ord).numpy()
    all_true_ord = torch.cat(all_true_ord).numpy()
    all_true_norm = torch.cat(all_true_norm).numpy()
    all_prompt_ids = torch.cat(all_prompt_ids).numpy()

    metrics = {}
    metrics['qwk_ordinal'] = float(cohen_kappa_score(all_true_ord, all_preds_ord, weights='quadratic'))
    preds_disc = np.clip(np.round(all_preds_reg * (num_classes - 1)).astype(int), 0, num_classes - 1)
    metrics['qwk_regression'] = float(cohen_kappa_score(all_true_ord, preds_disc, weights='quadratic'))
    metrics['qwk_best'] = max(metrics['qwk_ordinal'], metrics['qwk_regression'])
    metrics['rmse'] = float(np.sqrt(np.mean((all_preds_reg - all_true_norm) ** 2)))

    for pid in sorted(np.unique(all_prompt_ids)):
        mask = all_prompt_ids == pid
        if mask.sum() >= 2:
            metrics[f'qwk_prompt_{int(pid)}'] = float(
                cohen_kappa_score(all_true_ord[mask], all_preds_ord[mask], weights='quadratic')
            )

    return metrics

@torch.no_grad()
def generate_pseudo_labels(
    model, target_df, tokenizer, device, cfg, model_cfg,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.train()

    ds = build_dataset(target_df, tokenizer, model_cfg.max_length)
    loader = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    all_preds = []
    for mc_pass in tqdm(range(cfg.ust_mc_samples), desc="    MC passes", leave=False):
        batch_preds = []
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)

            with autocast(dtype=torch.float16, enabled=cfg.fp16):
                out = model(input_ids, attention_mask, global_attention_mask)
            batch_preds.append(out['reg_score'].cpu().numpy())
        all_preds.append(np.concatenate(batch_preds))

    all_preds = np.stack(all_preds)
    mean_preds = all_preds.mean(axis=0)
    uncertainties = all_preds.std(axis=0)

    threshold = np.percentile(uncertainties, cfg.ust_percentile)
    keep_mask = uncertainties <= threshold
    indices = np.where(keep_mask)[0]

    print(f"    MC Dropout: {cfg.ust_mc_samples} passes, "
          f"uncertainty threshold={threshold:.4f}, "
          f"keeping {len(indices)}/{len(mean_preds)} ({100*len(indices)/len(mean_preds):.1f}%)")

    return indices, mean_preds, uncertainties


def train_with_pseudo_labels(
    model, source_df, pseudo_target_df, val_loader,
    tokenizer, device, cfg, model_cfg, iteration, history,
) -> Dict[str, float]:
    model.train()

    src_ds = build_dataset(source_df, tokenizer, model_cfg.max_length)
    tgt_ds = build_dataset(pseudo_target_df, tokenizer, model_cfg.max_length)
    combined_ds = ConcatDataset([src_ds, tgt_ds])

    loader = DataLoader(
        combined_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )

    loss_fn = AESMultiTaskLoss(
        num_classes=model_cfg.num_classes,
        mse_weight=cfg.mse_weight,
        ranking_weight=cfg.ranking_weight,
        ordinal_weight=cfg.ordinal_weight,
        soft_qwk_weight=cfg.soft_qwk_weight,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.ust_lr, weight_decay=cfg.weight_decay,
    )
    scaler = GradScaler(enabled=cfg.fp16)

    epoch_losses = {}
    total_steps = 0

    for batch in tqdm(loader, desc=f"    UST iter {iteration}", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        norm_scores = batch['norm_score'].to(device)
        ordinal_labels = batch['ordinal_label'].to(device)
        has_label = batch['has_label'].to(device)

        with autocast(dtype=torch.float16, enabled=cfg.fp16):
            outputs = model(input_ids, attention_mask, global_attention_mask)
            losses = loss_fn(outputs, norm_scores, ordinal_labels, has_label)
            loss = losses['total'] / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (total_steps + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
        total_steps += 1

    avg = {k: v / max(total_steps, 1) for k, v in epoch_losses.items()}

    val_metrics = evaluate_da(model, val_loader, device, cfg, model_cfg.num_classes)
    avg['val_qwk'] = val_metrics['qwk_best']

    return avg

def run_stage_u(
    model,
    paths_da,
    paths: Paths,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    bin_edges: np.ndarray,
    tokenizer,
) -> Tuple:
    device = get_device()
    os.makedirs(paths_da['stage_u_dir'], exist_ok=True)
    os.makedirs(paths_da['graphs_dir'], exist_ok=True)
    history = AdaptationHistory()

    history.metadata = {
        'stage': 'U (DANN + CORAL + UST)',
        'dann_weight': train_cfg.dann_weight,
        'coral_weight': train_cfg.coral_weight,
        'ust_iterations': train_cfg.ust_iterations,
        'ust_mc_samples': train_cfg.ust_mc_samples,
        'ust_percentile': train_cfg.ust_percentile,
        'note': 'ASAP2 scores NEVER used. Only text for feature alignment.',
    }

    print("=" * 70)
    print("STAGE U: Domain Adaptation (DANN + CORAL + UST)")
    print("  ASAP2 essay text used for feature alignment only.")
    print("  ASAP2 scores are NEVER accessed during this stage.")
    print("=" * 70)
    print("\n[1/4] Loading datasets...")
    asap1_all, _ = load_asap1_labeled(paths, bin_edges=bin_edges, num_classes=model_cfg.num_classes)
    train_df, val_df = split_asap1_train_val(asap1_all, val_fraction=0.15, seed=train_cfg.seed)
    asap2_df = load_asap2(paths, bin_edges=bin_edges, num_classes=model_cfg.num_classes,
                          strip_source=True)

    asap2_adapt = asap2_df.copy()
    asap2_adapt['has_label'] = False
    print(f"  ASAP1 train: {len(train_df)} (labeled, for scoring losses)")
    print(f"  ASAP1 val:   {len(val_df)} (for model selection)")
    print(f"  ASAP2 adapt: {len(asap2_adapt)} (text only, has_label=False)")
    src_ds = build_dataset(train_df, tokenizer, model_cfg.max_length)
    tgt_ds = build_dataset(asap2_adapt, tokenizer, model_cfg.max_length)
    val_ds = build_dataset(val_df, tokenizer, model_cfg.max_length)

    src_loader = DataLoader(src_ds, batch_size=train_cfg.batch_size, shuffle=True,
                            num_workers=train_cfg.num_workers, pin_memory=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=train_cfg.batch_size, shuffle=True,
                            num_workers=train_cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
                            num_workers=train_cfg.num_workers, pin_memory=True)

    history.metadata['n_source'] = len(train_df)
    history.metadata['n_target'] = len(asap2_adapt)
    history.metadata['n_val'] = len(val_df)
    print(f"\n[2/4] DANN + CORAL training ({train_cfg.stage_u_epochs} epochs)...")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.ust_lr,
        weight_decay=train_cfg.weight_decay,
    )

    total_steps_u = (len(src_loader) // train_cfg.gradient_accumulation_steps) * train_cfg.stage_u_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps_u, 1),
    )

    scaler = GradScaler(enabled=train_cfg.fp16)
    coral_loss_fn = CORALLoss()
    best_qwk = -1.0
    patience_counter = 0
    global_step = 0

    for epoch in range(1, train_cfg.stage_u_epochs + 1):
        t0 = time.time()

        train_losses, global_step = train_dann_epoch(
            model, src_loader, tgt_loader, optimizer, scheduler,
            scaler, device, train_cfg, model_cfg,
            epoch, train_cfg.stage_u_epochs,
            history, global_step, coral_loss_fn,
        )

        val_metrics = evaluate_da(model, val_loader, device, train_cfg, model_cfg.num_classes)
        elapsed = time.time() - t0
        history.log_epoch(epoch, train_losses, val_metrics)

        print(f"  Epoch {epoch}/{train_cfg.stage_u_epochs} [{elapsed:.0f}s] "
              f"loss={train_losses['total']:.4f} "
              f"dann={train_losses['dann']:.4f} "
              f"coral={train_losses['coral']:.4f} | "
              f"val QWK={val_metrics['qwk_best']:.4f} "
              f"RMSE={val_metrics['rmse']:.4f}")

        if val_metrics['qwk_best'] > best_qwk:
            best_qwk = val_metrics['qwk_best']
            patience_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(paths_da['stage_u_dir'], "dann_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(
        os.path.join(paths_da['stage_u_dir'], "dann_best.pt"), map_location=device))

    print(f"\n  DANN+CORAL best val QWK: {best_qwk:.4f}")
    print(f"\n[3/4] Uncertainty-aware Self-Training ({train_cfg.ust_iterations} iterations)...")

    for ust_iter in range(1, train_cfg.ust_iterations + 1):
        print(f"\n  UST Iteration {ust_iter}/{train_cfg.ust_iterations}")
        pre_metrics = evaluate_da(model, val_loader, device, train_cfg, model_cfg.num_classes)
        qwk_before = pre_metrics['qwk_best']

        indices, mean_preds, uncertainties = generate_pseudo_labels(
            model, asap2_adapt, tokenizer, device, train_cfg, model_cfg,
        )

        if len(indices) < 10:
            print(f"    Too few confident samples ({len(indices)}). Stopping UST.")
            break

        pseudo_df = asap2_adapt.iloc[indices].copy()
        pseudo_df['norm_score'] = mean_preds[indices]

        pseudo_df['ordinal_label'] = pseudo_df['norm_score'].apply(
            lambda s: int(np.digitize(s, bin_edges))
        )

        pseudo_df['has_label'] = True

        ust_losses = train_with_pseudo_labels(
            model, train_df, pseudo_df, val_loader,
            tokenizer, device, train_cfg, model_cfg, ust_iter, history,
        )

        qwk_after = ust_losses['val_qwk']
        threshold = np.percentile(uncertainties, train_cfg.ust_percentile)

        history.log_ust_iteration(
            ust_iter, len(indices), uncertainties.mean(),
            threshold, qwk_before, qwk_after,
        )

        print(f"    QWK: {qwk_before:.4f} → {qwk_after:.4f} "
              f"({'↑' if qwk_after > qwk_before else '↓'} "
              f"{abs(qwk_after - qwk_before):.4f})")

        if qwk_after > best_qwk:
            best_qwk = qwk_after
            torch.save(model.state_dict(),
                       os.path.join(paths_da['stage_u_dir'], "ust_best.pt"))

    print(f"\n[4/4] Saving Stage U model...")
    best_ckpt = os.path.join(paths_da['stage_u_dir'], "ust_best.pt")

    if not os.path.exists(best_ckpt):
        best_ckpt = os.path.join(paths_da['stage_u_dir'], "dann_best.pt")

    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    final_path = os.path.join(paths_da['stage_u_dir'], "stage_u_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"  Final model: {final_path}")
    print(f"  Best val QWK: {best_qwk:.4f}")
    history.save(os.path.join(paths_da['graphs_dir'], "adaptation_history.json"))
    return model, history
