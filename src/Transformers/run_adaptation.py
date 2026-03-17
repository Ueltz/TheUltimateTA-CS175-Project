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

from config import (
    Paths, ModelConfig, LoRAConfig, TrainConfig,
    get_device, ASAP2_SCORE_RANGE,
)
from model_da import (
    AESDAModel, apply_lora_da, get_tokenizer,
    resize_embeddings, load_stage_s_into_da,
)
from data import (
    load_asap1_labeled, split_asap1_train_val, load_asap2,
    build_dataset, load_bin_edges, _load_asap1_unlabeled_tsv,
)
from train_adaptation import run_stage_u
from calibration import optimize_thresholds_qwk, apply_thresholds
from evaluate import full_evaluation_report

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_da_paths():
    return {
        'output_dir': './outputs_da',
        'stage_u_dir': './outputs_da/stage_u',
        'stage_c_dir': './outputs_da/stage_c',
        'graphs_dir': './outputs_da/graphs',
        'logs_dir': './outputs_da/logs',
    }

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
    parser = argparse.ArgumentParser(description="ASAP1→ASAP2 Domain Adaptation")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'u', 'eval'])
    parser.add_argument('--skip-ust', action='store_true',
                        help='Skip UST, only do DANN+CORAL')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-classes', type=int, default=6)
    parser.add_argument('--stage-s-checkpoint', type=str, default=None,
                        help='Path to Stage S checkpoint (default: outputs/stage_s/stage_s_final.pt)')
    args = parser.parse_args()

    paths = Paths()
    da_paths = get_da_paths()
    model_cfg = ModelConfig(max_length=args.max_length, num_classes=args.num_classes)
    lora_cfg = LoRAConfig(two_stage=True)
    train_cfg = TrainConfig(batch_size=args.batch_size, seed=args.seed)

    if args.skip_ust:
        train_cfg.ust_iterations = 0

    stage_s_ckpt = args.stage_s_checkpoint or os.path.join(paths.stage_s_dir, "stage_s_final.pt")

    print("=" * 70)
    print("ASAP1 → ASAP2 Domain Adaptation (DANN + CORAL + UST)")
    print("=" * 70)
    print(f"\n  Stage S checkpoint: {stage_s_ckpt}")
    print(f"  DA outputs:         {da_paths['output_dir']}/")
    print(f"  DANN + CORAL:       ASAP2 text only (scores never used)")
    print(f"  UST:                {'enabled' if not args.skip_ust else 'DISABLED'}")
    print(f"  Evaluation:         ALL of ASAP2 (separate from zero-shot)")

    set_seed(train_cfg.seed)
    device = get_device()
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    if not os.path.exists(stage_s_ckpt):
        print(f"\nERROR: Stage S checkpoint not found: {stage_s_ckpt}")
        print("Run the zero-shot pipeline first: python run_pipeline.py --stage s")
        sys.exit(1)

    bin_edges_path = os.path.join(paths.stage_s_dir, "bin_edges.json")
    if not os.path.exists(bin_edges_path):
        print(f"\nERROR: Bin edges not found: {bin_edges_path}")
        sys.exit(1)

    bin_edges = load_bin_edges(bin_edges_path)
    print(f"  Bin edges: {np.round(bin_edges, 4)}")

    for d in da_paths.values():
        os.makedirs(d, exist_ok=True)

    start_time = time.time()

    if args.stage in ['all', 'u']:
        print("\n" + "=" * 70)
        print("Loading Stage S checkpoint into DA model...")
        print("=" * 70)

        tokenizer = get_tokenizer(model_cfg)
        model = AESDAModel(model_cfg)
        resize_embeddings(model, tokenizer)

        if train_cfg.gradient_checkpointing:
            model.longformer.gradient_checkpointing_enable()

        model = apply_lora_da(model, lora_cfg)
        model = load_stage_s_into_da(model, stage_s_ckpt, device)
        model = model.to(device)

        model, history = run_stage_u(
            model, da_paths, paths, model_cfg, train_cfg, bin_edges, tokenizer,
        )

        print(f"\nStage U completed in {(time.time() - start_time) / 60:.1f} minutes")

    if args.stage in ['all', 'eval']:
        print("\n" + "=" * 70)
        print("EVALUATION: Domain-Adapted Model on ASAP2")
        print("=" * 70)

        tokenizer = get_tokenizer(model_cfg)

        if args.stage == 'eval':
            model = AESDAModel(model_cfg)
            resize_embeddings(model, tokenizer)
            model = apply_lora_da(model, lora_cfg)

            ckpt_path = os.path.join(da_paths['stage_u_dir'], "stage_u_final.pt")
            if not os.path.exists(ckpt_path):
                print(f"ERROR: No DA checkpoint at {ckpt_path}. Run --stage u first.")
                sys.exit(1)
            state_dict = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            print(f"  Loaded DA model from {ckpt_path}")
        else:
            model = model.to(device)

        print("\n[1/3] ASAP1 val inference (calibration)...")
        asap1_all, _ = load_asap1_labeled(paths, bin_edges=bin_edges, num_classes=model_cfg.num_classes)
        _, asap1_val_df = split_asap1_train_val(asap1_all, val_fraction=0.15, seed=train_cfg.seed)
        val_ds = build_dataset(asap1_val_df, tokenizer, model_cfg.max_length)
        val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size * 2,
                                shuffle=False, num_workers=train_cfg.num_workers, pin_memory=True)
        val_preds = predict_scores(model, val_loader, device, train_cfg.fp16)
        print(f"  ASAP1 val: {len(val_preds['reg_score'])} essays")

        print("\n[2/3] ASAP2 inference (ALL essays, stripped source texts)...")
        asap2_df = load_asap2(paths, bin_edges=bin_edges, num_classes=model_cfg.num_classes,
                              strip_source=True)
        test_ds = build_dataset(asap2_df, tokenizer, model_cfg.max_length)
        test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size * 2,
                                 shuffle=False, num_workers=train_cfg.num_workers, pin_memory=True)
        test_preds = predict_scores(model, test_loader, device, train_cfg.fp16)
        print(f"  ASAP2: {len(test_preds['reg_score'])} essays")

        print("\n[3/3] Calibration and evaluation...")
        lo, hi = ASAP2_SCORE_RANGE

        val_true_ord = val_preds['ordinal_label']
        val_pred_ord = val_preds['ordinal_labels']
        val_qwk = cohen_kappa_score(val_true_ord, val_pred_ord, weights='quadratic')
        print(f"\n  ASAP1 Val QWK (ordinal): {val_qwk:.4f}")

        test_true_native = test_preds['raw_score'].astype(int)
        test_preds_native = test_preds['reg_score'] * (hi - lo) + lo
        test_pred_ord = test_preds['ordinal_labels']

        qwk_ord = cohen_kappa_score(
            test_true_native, np.clip(test_pred_ord + lo, lo, hi), weights='quadratic'
        )

        test_disc_simple = np.clip(np.round(test_preds_native).astype(int), lo, hi)
        qwk_simple = cohen_kappa_score(test_true_native, test_disc_simple, weights='quadratic')

        thresholds, _ = optimize_thresholds_qwk(
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

        print(f"\n  ASAP2 Results (Domain-Adapted, ALL {len(test_true_native)} essays):")
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
            system_name=f"ASAP1→ASAP2 DA ({best_method})",
            save_path=os.path.join(da_paths['stage_c_dir'], "asap2_da_report.json"),
        )

        eval_data = {
            'model_type': 'domain_adapted',
            'asap2_true_scores': test_true_native.tolist(),
            'asap2_pred_reg': test_preds_native.tolist(),
            'asap2_pred_ordinal': (test_pred_ord + lo).tolist(),
            'asap2_pred_best': best_preds.astype(int).tolist(),
            'asap2_prompt_ids': test_preds['prompt_id'].tolist(),
            'asap2_qwk_ordinal': qwk_ord,
            'asap2_qwk_simple': qwk_simple,
            'asap2_qwk_threshold': qwk_thresh,
            'best_method': best_method,
            'best_qwk': best_qwk,
            'asap1_val_qwk': float(val_qwk),
            'bin_edges': bin_edges.tolist(),
        }

        for pid in sorted(np.unique(test_preds['prompt_id'])):
            mask = test_preds['prompt_id'] == pid
            if mask.sum() >= 2:
                qwk_p = cohen_kappa_score(
                    test_true_native[mask], best_preds[mask].astype(int), weights='quadratic'
                )
                eval_data[f'asap2_qwk_prompt_{int(pid)}'] = float(qwk_p)

        eval_path = os.path.join(da_paths['graphs_dir'], "evaluation_data_da.json")
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        print(f"\n  Evaluation data saved to {eval_path}")

        if os.path.exists(paths.asap1_unlabeled_test_tsv):
            print("\n  Generating ASAP1 test predictions (DA model)...")
            asap1_test_df = _load_asap1_unlabeled_tsv(paths.asap1_unlabeled_test_tsv)
            asap1_test_ds = build_dataset(asap1_test_df, tokenizer, model_cfg.max_length)
            asap1_test_loader = DataLoader(
                asap1_test_ds, batch_size=train_cfg.batch_size * 2, shuffle=False,
                num_workers=train_cfg.num_workers, pin_memory=True,
            )
            asap1_preds = predict_scores(model, asap1_test_loader, device, train_cfg.fp16)
            import pandas as pd
            pred_df = pd.DataFrame({
                'essay_id': asap1_test_df['essay_id'].values,
                'prompt_id': asap1_test_df['prompt_id'].values,
                'pred_reg_score': asap1_preds['reg_score'],
                'pred_ordinal_label': asap1_preds['ordinal_labels'],
            })
            pred_path = os.path.join(da_paths['stage_c_dir'], "asap1_test_predictions_da.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"  Saved {len(pred_df)} predictions to {pred_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"DA Pipeline completed in {total_time / 60:.1f} minutes")
    print(f"{'=' * 70}")
    print(f"\nData usage:")
    print(f"  ASAP1 train:  scoring losses (labeled)")
    print(f"  ASAP2 text:   feature alignment only (DANN+CORAL)")
    print(f"  ASAP2 scores: NEVER used during training")
    print(f"  UST:          pseudo-labels from model predictions only")
    print(f"\nOutputs (separate from zero-shot):")
    print(f"  DA model:       {da_paths['stage_u_dir']}/")
    print(f"  DA evaluation:  {da_paths['stage_c_dir']}/")
    print(f"  DA graph data:  {da_paths['graphs_dir']}/")
    print(f"\nCompare with zero-shot results in outputs/")


if __name__ == '__main__':
    main()
