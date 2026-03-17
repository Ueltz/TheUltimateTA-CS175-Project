"""
setup_data.py – Verify dataset setup and print statistics.
Shows averaged rater scores and per-prompt ranges.

    python setup_data.py
"""
import os
import pandas as pd

from config import Paths, ASAP1_RATER_RANGES, ASAP2_SCORE_RANGE


def main():
    paths = Paths()

    print("=" * 60)
    print("Dataset Verification (Averaged Rater Scores)")
    print("=" * 60)

    ok = True

    print(f"\n[ASAP1 Train] Checking {paths.asap1_train_tsv}...")
    if not os.path.exists(paths.asap1_train_tsv):
        print("  NOT FOUND!")
        ok = False
    else:
        df = pd.read_csv(paths.asap1_train_tsv, sep='\t', encoding='ISO-8859-1')
        print(f"  OK – {len(df)} essays loaded")

        print(f"\n  Per-prompt rater analysis:")
        print(f"  {'Prompt':>6} | {'N':>5} | {'Rater Range':>12} | "
              f"{'Avg Range':>12} | {'Avg Mean':>8} | {'R3 Present':>10}")
        print(f"  {'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}")

        for pid in sorted(df['essay_set'].unique()):
            sub = df[df['essay_set'] == pid]
            r1 = sub['rater1_domain1']
            r2 = sub['rater2_domain1']
            r3_count = sub['rater3_domain1'].notna().sum()
            avg = (r1 + r2) / 2.0
            lo, hi = ASAP1_RATER_RANGES[pid]

            print(f"  {pid:>6} | {len(sub):>5} | {int(r1.min()):>2}-{int(r1.max()):<8} | "
                  f"{avg.min():>4.1f}-{avg.max():<6.1f} | {avg.mean():>8.2f} | "
                  f"{r3_count:>10}")

            # Verify range matches config
            assert avg.min() >= lo, f"Prompt {pid}: avg min {avg.min()} < range lo {lo}"
            assert avg.max() <= hi, f"Prompt {pid}: avg max {avg.max()} > range hi {hi}"

        avg_words = df['essay'].str.split().str.len().mean()
        max_words = df['essay'].str.split().str.len().max()
        print(f"\n  Avg essay length: {avg_words:.0f} words")
        print(f"  Max essay length: {max_words} words")
        print(f"  All rater ranges verified against ASAP1_RATER_RANGES ✓")

    for name, path in [
        ("ASAP1 Valid (unlabeled)", paths.asap1_unlabeled_valid_tsv),
        ("ASAP1 Test (unlabeled)", paths.asap1_unlabeled_test_tsv),
    ]:
        print(f"\n[{name}] Checking {path}...")
        if os.path.exists(path):
            df = pd.read_csv(path, sep='\t', encoding='ISO-8859-1')
            print(f"  OK – {len(df)} essays (NO scores)")
            print(f"  Columns: {list(df.columns[:6])}")
        else:
            print(f"  Not found (optional)")

    print(f"\n[ASAP2] Checking {paths.asap2_train_csv}...")
    if not os.path.exists(paths.asap2_train_csv):
        print("  NOT FOUND!")
        ok = False
    else:
        df = pd.read_csv(paths.asap2_train_csv)
        print(f"  OK – {len(df)} essays (ALL held out for testing)")

        scores = df['score'].dropna()
        lo, hi = ASAP2_SCORE_RANGE
        print(f"  Score range: {int(scores.min())}-{int(scores.max())} "
              f"(expected {lo}-{hi})")
        print(f"  Score distribution:")
        for s in sorted(scores.unique()):
            n = (scores == s).sum()
            pct = 100 * n / len(scores)
            print(f"    Score {int(s)}: {n:>5} ({pct:>5.1f}%)")

        prompts = sorted(df['prompt_name'].unique())
        print(f"\n  Prompts ({len(prompts)}):")
        for p in prompts:
            n = (df['prompt_name'] == p).sum()
            print(f"    {p}: {n} essays")

        source_cols = [c for c in df.columns if c.startswith('source_text_')]
        print(f"\n  Source text columns: {source_cols}")
        for sc in source_cols:
            non_na = df[sc].apply(
                lambda x: isinstance(x, str) and x.strip().upper() != "NA"
            ).sum()
            print(f"    {sc}: {non_na}/{len(df)} non-empty")

    print(f"\n{'=' * 60}")
    print("Hardware Check")
    print(f"{'=' * 60}")
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu}")
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("  WARNING: No GPU detected!")

    print(f"\n{'=' * 60}")
    if ok:
        print("All required datasets found! Ready to run:")
        print("  python run_pipeline.py")
        print("\nData flow:")
        print("  TRAIN on:  ASAP1 training_set_rel3.tsv (averaged raters, 85/15 split)")
        print("  TEST on:   ALL of ASAP2 (zero leakage)")
        print("  ASAP1 test: predictions only (no ground truth scores)")
    else:
        print("Some required datasets missing. Fix paths and retry.")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
