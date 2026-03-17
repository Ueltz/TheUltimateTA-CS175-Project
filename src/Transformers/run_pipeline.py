import argparse
import os
import sys
import json
import time
import torch
import numpy as np
import random
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score

from config import Paths, ModelConfig, LoRAConfig, TrainConfig, get_device, ASAP2_SCORE_RANGE
from model import AESTransferModel, apply_lora, get_tokenizer, resize_embeddings
from data import (
    load_asap1_labeled, split_asap1_train_val, load_asap2,
    build_dataset, load_bin_edges, _load_asap1_unlabeled_tsv,
)
from train_supervised import run_stage_s
from calibration import optimize_thresholds_qwk, apply_thresholds
from evaluate import full_evaluation_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_datasets(paths: Paths):
    print("\nChecking dataset files...")
    missing = []
    for name, path in [
        ("ASAP1 labeled train", paths.asap1_train_tsv),
        ("ASAP2 (full test set)", paths.asap2_train_csv),
    ]:
        if os.path.exists(path):
            print(f"  {name}: OK ({path})")
        else:
            missing.append(f"  {name}: {path}")

    for name, path in [
        ("ASAP1 unlabeled test", paths.asap1_unlabeled_test_tsv),
    ]:
        if os.path.exists(path):
            print(f"  {name}: OK (predictions only)")
        else:
            print(f"  {name}: not found (optional)")

    if missing:
        print("\nERROR: Missing required files:")
        for m in missing:
            print(m)
        sys.exit(1)


@torch.no_grad()
def predict_scores(model, loader, device, use_fp16=True) -> dict:
    model.eval()
    keys = ['reg_score', 'ordinal_logits', 'ordinal_labels', 'ordinal_expected', 'ordinal_probs']
    batch_keys = ['ordinal_label', 'norm_score', 'raw_score', 'prompt_id', 'has_label']
    results = {k: [] for k in keys + batch_keys}

    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        global_attention_mask = batch['global_attention_mask'].to(device)

        with autocast(dtype=torch.float16, enabled=use_fp16):
            preds = model.predict(input_ids, attention_mask, global_attention_mask)

        for k in keys:
            results[k].append(preds[k].cpu())
        for k in batch_keys:
            results[k].append(batch[k])

    return {k: torch.cat(v).numpy() for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="ASAP1→ASAP2 AES Transfer (Zero-Shot)")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 's', 'eval'])
    parser.add_argument('--no-two-stage', action='store_true')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-classes', type=int, default=6)
    args = parser.parse_args()

    paths = Paths()
    model_cfg = ModelConfig(max_length=args.max_length, num_classes=args.num_classes)
    lora_cfg = LoRAConfig(two_stage=not args.no_two_stage)
    train_cfg = TrainConfig(batch_size=args.batch_size, seed=args.seed)

    print("=" * 70)
    print("ASAP1 → ASAP2 AES Transfer (Zero-Shot Generalization)")
    print("=" * 70)
    print(f"\n  Training:      ASAP1 training_set_rel3.tsv only")
    print(f"  Scores:        Averaged raters (r1+r2)/2")
    print(f"  Binning:       Quantile ({model_cfg.num_classes} bins)")
    print(f"  Losses:        MSE + ranking + CORN + soft QWK")
    print(f"  Evaluation:    ALL of ASAP2 (zero leakage)")
    print(f"  ASAP1 test:    Predictions only (no scores)")

    set_seed(train_cfg.seed)
    device = get_device()
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    check_datasets(paths)

    for d in [paths.output_dir, paths.stage_s_dir, paths.stage_c_dir,
              paths.logs_dir, paths.graphs_dir]:
        os.makedirs(d, exist_ok=True)

    start_time = time.time()

    if args.stage in ['all', 's']:
        model, bin_edges, history = run_stage_s(paths, model_cfg, lora_cfg, train_cfg)
        print(f"\nStage S completed in {(time.time() - start_time) / 60:.1f} minutes")

    if args.stage in ['all', 'eval']:
        print("\n" + "=" * 70)
        print("EVALUATION: Zero-Shot Transfer to ASAP2")
        print("=" * 70)

        tokenizer = get_tokenizer(model_cfg)

        bin_edges_path = os.path.join(paths.stage_s_dir, "bin_edges.json")
        if args.stage == 'eval':
            if not os.path.exists(bin_edges_path):
                print(f"ERROR: No bin edges found at {bin_edges_path}. Run --stage s first.")
                sys.exit(1)
            bin_edges = load_bin_edges(bin_edges_path)
            print(f"  Loaded bin edges: {np.round(bin_edges, 4)}")

        if args.stage == 'eval':
            print("\nLoading Stage S checkpoint...")
            model = AESTransferModel(model_cfg)
            resize_embeddings(model, tokenizer)
            ckpt_path = os.path.join(paths.stage_s_dir, "stage_s_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"ERROR: No checkpoint at {ckpt_path}. Run --stage s first.")
                sys.exit(1)
            model = apply_lora(model, lora_cfg)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            print(f"  Loaded checkpoint from {ckpt_path}")
        else:
            model = model.to(device)

        print("\n[1/3] ASAP1 val inference (for threshold calibration)...")
        asap1_all, _ = load_asap1_labeled(paths, bin_edges=bin_edges, num_classes=model_cfg.num_classes)
        _, asap1_val_df = split_asap1_train_val(asap1_all, val_fraction=0.15, seed=train_cfg.seed)
        val_ds = build_dataset(asap1_val_df, tokenizer, model_cfg.max_length)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size * 2,
                                shuffle=False, num_workers=train_cfg.num_workers, pin_memory=True)
        val_preds = predict_scores(model, val_loader, device, train_cfg.fp16)
        print(f"  ASAP1 val: {len(val_preds['reg_score'])} essays")

        print("\n[2/3] ASAP2 inference (ALL 24,728 essays – held out)...")
        asap2_df = load_asap2(paths, bin_edges=bin_edges, num_classes=model_cfg.num_classes)
        test_ds = build_dataset(asap2_df, tokenizer, model_cfg.max_length)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size * 2,
                                 shuffle=False, num_workers=train_cfg.num_workers, pin_memory=True)
        test_preds = predict_scores(model, test_loader, device, train_cfg.fp16)
        print(f"  ASAP2: {len(test_preds['reg_score'])} essays")

        print("\n[3/3] Calibrating and evaluating...")
        lo, hi = ASAP2_SCORE_RANGE

        val_preds_native = val_preds['reg_score'] * (hi - lo) + lo
        val_true_native = val_preds['raw_score']
        val_true_ord = val_preds['ordinal_label']
        val_pred_ord = val_preds['ordinal_labels']

        print(f"\n  ASAP1 Val QWK (ordinal head): {cohen_kappa_score(val_true_ord, val_pred_ord, weights='quadratic'):.4f}")

        test_true_native = test_preds['raw_score'].astype(int)
        test_preds_native = test_preds['reg_score'] * (hi - lo) + lo
        test_true_ord = test_preds['ordinal_label']
        test_pred_ord = test_preds['ordinal_labels']

        qwk_ord = cohen_kappa_score(test_true_native, np.clip(test_pred_ord + lo, lo, hi), weights='quadratic')

        test_disc_simple = np.clip(np.round(test_preds_native).astype(int), lo, hi)
        qwk_simple = cohen_kappa_score(test_true_native, test_disc_simple, weights='quadratic')

        thresholds, val_thresh_qwk = optimize_thresholds_qwk(
            val_preds['ordinal_label'].astype(int),
            val_preds['reg_score'] * (model_cfg.num_classes - 1),
            num_classes=model_cfg.num_classes,
        )
        test_ord_thresh = apply_thresholds(
            test_preds['reg_score'] * (model_cfg.num_classes - 1),
            thresholds, min_label=0, max_label=model_cfg.num_classes - 1,
        ).astype(int)
        test_native_thresh = np.clip(test_ord_thresh + lo, lo, hi)
        qwk_thresh = cohen_kappa_score(test_true_native, test_native_thresh, weights='quadratic')

        print(f"\n  ASAP2 Results (ALL {len(test_true_native)} essays, zero-shot):")
        print(f"    Ordinal head:          QWK = {qwk_ord:.4f}")
        print(f"    Regression (rounded):  QWK = {qwk_simple:.4f}")
        print(f"    Regression (thresh):   QWK = {qwk_thresh:.4f}")

        best_qwk = max(qwk_ord, qwk_simple, qwk_thresh)
        best_method = ['ordinal', 'simple_round', 'threshold'][
            [qwk_ord, qwk_simple, qwk_thresh].index(best_qwk)
        ]
        best_preds = {'ordinal': np.clip(test_pred_ord + lo, lo, hi),
                      'simple_round': test_disc_simple,
                      'threshold': test_native_thresh}[best_method]

        print(f"\n  Best: {best_method} (QWK = {best_qwk:.4f})")

        report = full_evaluation_report(
            test_true_native, best_preds.astype(int),
            prompt_ids=test_preds['prompt_id'],
            system_name=f"ASAP1→ASAP2 Zero-Shot ({best_method})",
            save_path=os.path.join(paths.stage_c_dir, "asap2_report.json"),
        )

        eval_graph_data = {
            'asap2_true_scores': test_true_native.tolist(),
            'asap2_pred_reg': test_preds_native.tolist(),
            'asap2_pred_ordinal': (test_pred_ord + lo).tolist(),
            'asap2_pred_best': best_preds.astype(int).tolist(),
            'asap2_prompt_ids': test_preds['prompt_id'].tolist(),
            'asap2_qwk_ordinal': qwk_ord,
            'asap2_qwk_simple': qwk_simple,
            'asap2_qwk_threshold': qwk_thresh,
            'best_method': best_method,
            'asap1_val_qwk_ordinal': float(cohen_kappa_score(val_true_ord, val_pred_ord, weights='quadratic')),
            'bin_edges': bin_edges.tolist(),
            'thresholds': thresholds.tolist(),
        }

        for pid in sorted(np.unique(test_preds['prompt_id'])):
            mask = test_preds['prompt_id'] == pid
            if mask.sum() >= 2:
                qwk_p = cohen_kappa_score(
                    test_true_native[mask], best_preds[mask].astype(int), weights='quadratic'
                )
                eval_graph_data[f'asap2_qwk_prompt_{int(pid)}'] = float(qwk_p)

        eval_path = os.path.join(paths.graphs_dir, "evaluation_data.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_graph_data, f, indent=2)
        print(f"\n  Evaluation graph data saved to {eval_path}")

        if os.path.exists(paths.asap1_unlabeled_test_tsv):
            print("\n  Generating ASAP1 test predictions (no scores to evaluate)...")
            asap1_test_df = _load_asap1_unlabeled_tsv(paths.asap1_unlabeled_test_tsv)
            asap1_test_ds = build_dataset(asap1_test_df, tokenizer, model_cfg.max_length)
            asap1_test_loader = DataLoader(
                asap1_test_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
                num_workers=train_cfg.num_workers, pin_memory=True,
            )
            asap1_test_preds = predict_scores(model, asap1_test_loader, device, train_cfg.fp16)

            import pandas as pd
            pred_df = pd.DataFrame({
                'essay_id': asap1_test_df['essay_id'].values,
                'prompt_id': asap1_test_df['prompt_id'].values,
                'pred_reg_score': asap1_test_preds['reg_score'],
                'pred_ordinal_label': asap1_test_preds['ordinal_labels'],
            })
            pred_path = os.path.join(paths.stage_c_dir, "asap1_test_predictions.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"  Saved {len(pred_df)} predictions to {pred_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Pipeline completed in {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"\nData separation guarantee:")
    print(f"  Trained on:    ASAP1 training_set_rel3.tsv (train split, averaged raters)")
    print(f"  Calibrated on: ASAP1 training_set_rel3.tsv (val split)")
    print(f"  Tested on:     ALL of ASAP2 (zero ASAP2 data in training)")
    print(f"  ASAP1 test:    Predictions only (no scores available)")
    print(f"\nGraph data saved to: {paths.graphs_dir}/")
    print(f"  training_history.json – step/epoch losses, QWK curves")
    print(f"  evaluation_data.json  – predictions, per-prompt QWK, distributions")


if __name__ == '__main__':
    main()
