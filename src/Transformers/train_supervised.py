import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
from typing import Dict, Tuple, List

from config import TrainConfig, ModelConfig, LoRAConfig, Paths, get_device
from model import (
    AESTransferModel, apply_lora, freeze_backbone_except_top,
    unfreeze_all, get_tokenizer, resize_embeddings,
)
from losses import AESMultiTaskLoss
from data import (
    build_dataset, load_asap1_labeled, split_asap1_train_val,
    save_bin_edges,
)



class TrainingHistory:

    def __init__(self):
        self.step_logs: List[dict] = []
        self.epoch_logs: List[dict] = []
        self.metadata: dict = {}

    def log_step(self, step: int, epoch: int, stage: str, losses: dict, lr: float):
        entry = {
            'global_step': step,
            'epoch': epoch,
            'stage': stage,
            'lr': lr,
        }
        for k, v in losses.items():
            entry[f'loss_{k}'] = float(v) if not isinstance(v, float) else v
        self.step_logs.append(entry)

    def log_epoch(self, epoch: int, stage: str, train_metrics: dict, val_metrics: dict):
        entry = {
            'epoch': epoch,
            'stage': stage,
        }
        for k, v in train_metrics.items():
            entry[f'train_{k}'] = float(v)
        for k, v in val_metrics.items():
            entry[f'val_{k}'] = float(v)
        self.epoch_logs.append(entry)

    def save(self, path: str):
        data = {
            'metadata': self.metadata,
            'step_logs': self.step_logs,
            'epoch_logs': self.epoch_logs,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Training history saved to {path} "
              f"({len(self.step_logs)} steps, {len(self.epoch_logs)} epochs)")

def train_one_epoch(
    model: AESTransferModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: AESMultiTaskLoss,
    scaler: GradScaler,
    device: torch.device,
    cfg: TrainConfig,
    grad_accum_steps: int,
    history: TrainingHistory,
    epoch: int,
    stage: str,
    global_step: int,
) -> Tuple[Dict[str, float], int]:
    model.train()
    epoch_losses = {}
    total_steps = 0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc="  Training", leave=False)

    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        norm_scores = batch['norm_score'].to(device)
        ordinal_labels = batch['ordinal_label'].to(device)
        has_label = batch['has_label'].to(device)

        with autocast(dtype=torch.float16, enabled=cfg.fp16):
            outputs = model(input_ids, attention_mask, global_attention_mask)
            losses = loss_fn(outputs, norm_scores, ordinal_labels, has_label)
            loss = losses['total'] / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
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
                global_step, epoch, stage,
                {k: v.item() for k, v in losses.items()},
                lr,
            )

        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'mse': f"{losses['mse'].item():.4f}",
            'sqwk': f"{losses['soft_qwk'].item():.4f}",
        })

    avg = {k: v / max(total_steps, 1) for k, v in epoch_losses.items()}
    return avg, global_step

@torch.no_grad()
def evaluate(
    model: AESTransferModel,
    loader: DataLoader,
    loss_fn: AESMultiTaskLoss,
    device: torch.device,
    cfg: TrainConfig,
    num_classes: int = 6,
) -> Dict[str, float]:
    model.eval()
    all_preds_reg = []
    all_preds_ord = []
    all_true_norm = []
    all_true_ord = []
    all_prompt_ids = []
    epoch_losses = {}
    total_steps = 0

    for batch in tqdm(loader, desc="  Evaluating", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)
        norm_scores = batch['norm_score'].to(device)
        ordinal_labels = batch['ordinal_label'].to(device)
        has_label = batch['has_label'].to(device)

        with autocast(dtype=torch.float16, enabled=cfg.fp16):
            outputs = model(input_ids, attention_mask, global_attention_mask)
            losses = loss_fn(outputs, norm_scores, ordinal_labels, has_label)

        preds = model.predict(input_ids, attention_mask, global_attention_mask)
        all_preds_reg.append(preds['reg_score'].cpu())
        all_preds_ord.append(preds['ordinal_labels'].cpu())
        all_true_norm.append(norm_scores.cpu())
        all_true_ord.append(ordinal_labels.cpu())
        all_prompt_ids.append(batch['prompt_id'])

        for k, v in losses.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

        total_steps += 1

    metrics = {k: v / max(total_steps, 1) for k, v in epoch_losses.items()}
    all_preds_reg = torch.cat(all_preds_reg).numpy()
    all_preds_ord = torch.cat(all_preds_ord).numpy()
    all_true_norm = torch.cat(all_true_norm).numpy()
    all_true_ord = torch.cat(all_true_ord).numpy()
    all_prompt_ids = torch.cat(all_prompt_ids).numpy()

    metrics['qwk_ordinal'] = float(cohen_kappa_score(
        all_true_ord, all_preds_ord, weights='quadratic'
    ))

    preds_disc = np.clip(np.round(all_preds_reg * (num_classes - 1)).astype(int), 0, num_classes - 1)
    metrics['qwk_regression'] = float(cohen_kappa_score(
        all_true_ord, preds_disc, weights='quadratic'
    ))

    metrics['qwk_best'] = max(metrics['qwk_ordinal'], metrics['qwk_regression'])
    metrics['rmse'] = float(np.sqrt(np.mean((all_preds_reg - all_true_norm) ** 2)))

    for pid in sorted(np.unique(all_prompt_ids)):
        mask = all_prompt_ids == pid

        if mask.sum() >= 2:
            qwk_p = cohen_kappa_score(all_true_ord[mask], all_preds_ord[mask], weights='quadratic')
            metrics[f'qwk_prompt_{int(pid)}'] = float(qwk_p)

    metrics['pred_reg_mean'] = float(all_preds_reg.mean())
    metrics['pred_reg_std'] = float(all_preds_reg.std())
    metrics['true_norm_mean'] = float(all_true_norm.mean())
    metrics['true_norm_std'] = float(all_true_norm.std())

    for k in range(num_classes):
        metrics[f'pred_ord_bin_{k}_pct'] = float(np.mean(all_preds_ord == k) * 100)
        metrics[f'true_ord_bin_{k}_pct'] = float(np.mean(all_true_ord == k) * 100)

    return metrics



def run_stage_s(
    paths: Paths,
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig,
    train_cfg: TrainConfig,
) -> Tuple[AESTransferModel, np.ndarray, TrainingHistory]:
    device = get_device()
    os.makedirs(paths.stage_s_dir, exist_ok=True)
    os.makedirs(paths.graphs_dir, exist_ok=True)
    history = TrainingHistory()

    history.metadata = {
        'stage': 'S',
        'backbone': model_cfg.backbone,
        'max_length': model_cfg.max_length,
        'num_classes': model_cfg.num_classes,
        'lora_r': lora_cfg.r,
        'two_stage': lora_cfg.two_stage,
        'batch_size': train_cfg.batch_size,
        'grad_accum': train_cfg.gradient_accumulation_steps,
        'effective_batch': train_cfg.batch_size * train_cfg.gradient_accumulation_steps,
        'score_method': 'averaged_raters_(r1+r2)/2',
        'binning': 'quantile',
        'losses': ['mse', 'pairwise_ranking', 'corn_ordinal', 'soft_qwk'],
    }

    print("=" * 70)
    print("STAGE S: Supervised Training on ASAP1")
    print("  Scores: averaged raters (r1+r2)/2, normalized [0,1]")
    print("  Binning: quantile-based ordinal bins")
    print("  Losses: MSE + ranking + CORN + soft QWK")
    print("=" * 70)
    print("\n[1/5] Loading ASAP1 data (averaged rater scores)...")
    tokenizer = get_tokenizer(model_cfg)
    asap1_all, bin_edges = load_asap1_labeled(paths, num_classes=model_cfg.num_classes)
    train_df, val_df = split_asap1_train_val(asap1_all, val_fraction=0.15, seed=train_cfg.seed)
    print(f"  Total labeled: {len(asap1_all)} essays")
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    save_bin_edges(bin_edges, os.path.join(paths.stage_s_dir, "bin_edges.json"))
    history.metadata['n_train'] = len(train_df)
    history.metadata['n_val'] = len(val_df)
    history.metadata['bin_edges'] = bin_edges.tolist()
    train_ds = build_dataset(train_df, tokenizer, model_cfg.max_length)
    val_ds = build_dataset(val_df, tokenizer, model_cfg.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        num_workers=train_cfg.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
        num_workers=train_cfg.num_workers, pin_memory=True,
    )

    print("\n[2/5] Building model...")
    model = AESTransferModel(model_cfg)
    resize_embeddings(model, tokenizer)

    if train_cfg.gradient_checkpointing:
        model.longformer.gradient_checkpointing_enable()

    model = model.to(device)

    loss_fn = AESMultiTaskLoss(
        num_classes=model_cfg.num_classes,
        mse_weight=train_cfg.mse_weight,
        ranking_weight=train_cfg.ranking_weight,
        ordinal_weight=train_cfg.ordinal_weight,
        soft_qwk_weight=train_cfg.soft_qwk_weight,
    )

    scaler = GradScaler(enabled=train_cfg.fp16)
    global_step = 0

    if lora_cfg.two_stage:
        print(f"\n[3/5] Two-Stage LoRA – Stage 1: top {lora_cfg.stage1_unfreeze_layers} layers...")
        freeze_backbone_except_top(model, lora_cfg.stage1_unfreeze_layers)

        optimizer_s1 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_cfg.stage_s_lr,
            weight_decay=train_cfg.weight_decay,
        )

        s1_epochs = max(1, train_cfg.stage_s_epochs // 2)
        total_steps_s1 = (len(train_loader) // train_cfg.gradient_accumulation_steps) * s1_epochs

        scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_s1, T_max=max(total_steps_s1, 1),
        )

        best_qwk = -1.0
        patience_counter = 0

        for epoch in range(s1_epochs):
            print(f"\n  Stage 1 Epoch {epoch + 1}/{s1_epochs}")
            t0 = time.time()

            train_losses, global_step = train_one_epoch(
                model, train_loader, optimizer_s1, scheduler_s1,
                loss_fn, scaler, device, train_cfg,
                train_cfg.gradient_accumulation_steps,
                history, epoch + 1, "S1", global_step,
            )

            val_metrics = evaluate(model, val_loader, loss_fn, device, train_cfg, model_cfg.num_classes)
            elapsed = time.time() - t0
            history.log_epoch(epoch + 1, "S1", train_losses, val_metrics)

            print(f"  [{elapsed:.0f}s] Train loss={train_losses['total']:.4f} | "
                  f"Val loss={val_metrics['total']:.4f} | "
                  f"QWK(ord)={val_metrics['qwk_ordinal']:.4f} | "
                  f"QWK(reg)={val_metrics['qwk_regression']:.4f} | "
                  f"RMSE={val_metrics['rmse']:.4f}")

            if val_metrics['qwk_best'] > best_qwk:
                best_qwk = val_metrics['qwk_best']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(paths.stage_s_dir, "stage1_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(torch.load(
            os.path.join(paths.stage_s_dir, "stage1_best.pt"), map_location=device))

        print(f"\n  Stage 1 best QWK: {best_qwk:.4f}")
        print(f"\n[4/5] Two-Stage LoRA – Stage 2: LoRA adapters...")
        unfreeze_all(model)
        model = apply_lora(model, lora_cfg)
        model = model.to(device)

        optimizer_s2 = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_cfg.stage_s_lora_lr,
            weight_decay=train_cfg.weight_decay,
        )

        s2_epochs = train_cfg.stage_s_epochs - s1_epochs
        total_steps_s2 = (len(train_loader) // train_cfg.gradient_accumulation_steps) * s2_epochs

        scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_s2, T_max=max(total_steps_s2, 1),
        )

        best_qwk = -1.0
        patience_counter = 0

        for epoch in range(s2_epochs):
            print(f"\n  Stage 2 Epoch {epoch + 1}/{s2_epochs}")
            t0 = time.time()

            train_losses, global_step = train_one_epoch(
                model, train_loader, optimizer_s2, scheduler_s2,
                loss_fn, scaler, device, train_cfg,
                train_cfg.gradient_accumulation_steps,
                history, s1_epochs + epoch + 1, "S2", global_step,
            )

            val_metrics = evaluate(model, val_loader, loss_fn, device, train_cfg, model_cfg.num_classes)
            elapsed = time.time() - t0
            history.log_epoch(s1_epochs + epoch + 1, "S2", train_losses, val_metrics)

            print(f"  [{elapsed:.0f}s] Train loss={train_losses['total']:.4f} | "
                  f"Val loss={val_metrics['total']:.4f} | "
                  f"QWK(ord)={val_metrics['qwk_ordinal']:.4f} | "
                  f"QWK(reg)={val_metrics['qwk_regression']:.4f} | "
                  f"RMSE={val_metrics['rmse']:.4f}")

            if val_metrics['qwk_best'] > best_qwk:
                best_qwk = val_metrics['qwk_best']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(paths.stage_s_dir, "stage2_best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(torch.load(
            os.path.join(paths.stage_s_dir, "stage2_best.pt"), map_location=device))

        print(f"\n  Stage 2 best QWK: {best_qwk:.4f}")

    else:
        print(f"\n[3/5] Single-stage LoRA...")
        model = apply_lora(model, lora_cfg)
        model = model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=train_cfg.stage_s_lora_lr,
            weight_decay=train_cfg.weight_decay,
        )

        total_steps = (len(train_loader) // train_cfg.gradient_accumulation_steps) * train_cfg.stage_s_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1),
        )

        best_qwk = -1.0
        patience_counter = 0

        for epoch in range(train_cfg.stage_s_epochs):
            print(f"\n  Epoch {epoch + 1}/{train_cfg.stage_s_epochs}")
            t0 = time.time()

            train_losses, global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler,
                loss_fn, scaler, device, train_cfg,
                train_cfg.gradient_accumulation_steps,
                history, epoch + 1, "S", global_step,
            )

            val_metrics = evaluate(model, val_loader, loss_fn, device, train_cfg, model_cfg.num_classes)
            elapsed = time.time() - t0
            history.log_epoch(epoch + 1, "S", train_losses, val_metrics)

            print(f"  [{elapsed:.0f}s] Train loss={train_losses['total']:.4f} | "
                  f"Val loss={val_metrics['total']:.4f} | "
                  f"QWK(ord)={val_metrics['qwk_ordinal']:.4f} | "
                  f"QWK(reg)={val_metrics['qwk_regression']:.4f}")

            if val_metrics['qwk_best'] > best_qwk:
                best_qwk = val_metrics['qwk_best']
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(paths.stage_s_dir, "best.pt"))
            else:
                patience_counter += 1
                if patience_counter >= train_cfg.patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(torch.load(
            os.path.join(paths.stage_s_dir, "best.pt"), map_location=device))
        print(f"\n  Best QWK: {best_qwk:.4f}")

    print(f"\n[5/5] Saving...")
    torch.save(model.state_dict(), os.path.join(paths.stage_s_dir, "stage_s_final.pt"))
    tokenizer.save_pretrained(os.path.join(paths.stage_s_dir, "tokenizer"))
    history.save(os.path.join(paths.graphs_dir, "training_history.json"))
    return model, bin_edges, history
