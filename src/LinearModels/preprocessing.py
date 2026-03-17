"""
Use this file so we have identical:
  - Rater averaging (ASAP1: (rater1 + rater2) / 2)
  - Per-prompt normalization to [0, 1]
  - Train/val split (same seed, same stratification)
  - ASAP2 loading and normalization
  - Text preprocessing

Your model code just needs to vectorize the 'text' column however you want
and predict 'norm_score' (or 'raw_score' if you prefer the original scale).

Usage:
    from shared_data_prep import load_all_data

    data = load_all_data(
        asap1_path='./data/training_set_rel3.tsv',
        asap2_path='./data/ASAP2_train_sourcetexts.csv',
    )

    data['asap1_train']    # DataFrame – labeled ASAP1 training essays (85%)
    data['asap1_val']      # DataFrame – labeled ASAP1 validation essays (15%)
    data['asap2']          # DataFrame – ALL ASAP2 essays (held-out test)
    data['config']         # dict – all parameters used

    # Every DataFrame has these columns:
    #   essay_id    – unique ID
    #   prompt_id   – integer prompt identifier
    #   text        – preprocessed essay text (model input)
    #   raw_score   – original score (averaged raters for ASAP1, 1-6 for ASAP2)
    #   norm_score  – normalized to [0, 1] using per-prompt ranges

Requirements:
    pip install pandas numpy scikit-learn
"""
import math

import pandas as pd
import numpy as np
import re
import json
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Dict, Tuple

from torch import tensor


# ═══════════════════════════════════════════════════════════════════════════
# FIXED CONSTANTS – DO NOT CHANGE (ensures consistency across team)
# ═══════════════════════════════════════════════════════════════════════════

# ASAP1: per-rater scoring ranges (verified from data)
# Scores are averaged: (rater1_domain1 + rater2_domain1) / 2
ASAP1_RATER_RANGES = {
    1: (1, 6),    # persuasive: effects of computers
    2: (1, 6),    # persuasive: censorship in libraries
    3: (0, 3),    # source-based: mood in memoir
    4: (0, 3),    # source-based: obstacles builders faced
    5: (0, 4),    # source-based: setting from narrator's perspective
    6: (0, 4),    # source-based: challenges of space travel
    7: (0, 12),   # narrative: patience
    8: (0, 30),   # narrative: laughter
}

# ASAP2: uniform 1–6 scoring
ASAP2_SCORE_RANGE = (1, 6)

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 42

# Short prompt descriptions (used as text prefix)
ASAP1_PROMPT_DESCRIPTIONS = {
    1: "Write about the effects computers have on people.",
    2: "Should you censor content in libraries?",
    3: "Source-based: describe the mood created by the author in the memoir.",
    4: "Source-based: based on the excerpt, describe the obstacles the builders faced.",
    5: "Source-based: describe the features of the setting from the narrator's perspective.",
    6: "Source-based: based on the excerpt, describe the challenges of space travel.",
    7: "Write about patience. Tell a story about a time when you were patient.",
    8: "Write about laughter's role in life from a personal experience.",
}

ASAP2_PROMPT_DESCRIPTIONS = {
    '"A Cowboy Who Rode the Waves"': "Source-based: describe the narrator's perspective and experiences.",
    'Car-free cities': "Write about the advantages or disadvantages of limiting car usage in cities.",
    'Does the electoral college work?': "Write about whether the electoral college system is effective.",
    'Driverless cars': "Write about the development and implications of driverless car technology.",
    'Exploring Venus': "Write about the challenges and value of exploring the planet Venus.",
    'Facial action coding system': "Write about the value of technology that reads facial expressions.",
    'The Face on Mars': "Write about the landform on Mars known as the Face and what it represents.",
}


# ═══════════════════════════════════════════════════════════════════════════
# Text preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_essay(text: str) -> str:
    """Clean essay text"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', ' [PARA] ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_text_input(essay_text: str, prompt_description: str = "") -> str:
    """
    Build the model input text.
    Format: [prompt description] [SEP] [essay text]

    If your model uses a different separator token, do a string replace
    on [SEP] after calling this function.
    """
    essay_text = preprocess_essay(essay_text)
    if prompt_description:
        prompt_description = preprocess_essay(prompt_description)
        return f"{prompt_description} [SEP] {essay_text}"
    return essay_text


# ═══════════════════════════════════════════════════════════════════════════
# Score normalization
# ═══════════════════════════════════════════════════════════════════════════

def normalize_asap1(avg_score: float, prompt_id: int) -> float:
    """Normalize averaged ASAP1 rater score to [0, 1]."""
    lo, hi = ASAP1_RATER_RANGES[prompt_id]
    if hi == lo:
        return 0.5
    return (avg_score - lo) / (hi - lo)


def denormalize_asap1(norm_score: float, prompt_id: int) -> float:
    """Convert [0, 1] back to ASAP1 averaged rater scale."""
    lo, hi = ASAP1_RATER_RANGES[prompt_id]
    return round(norm_score * (hi - lo) + lo)


def normalize_asap2(score: float) -> float:
    """Normalize ASAP2 score (1-6) to [0, 1]."""
    lo, hi = ASAP2_SCORE_RANGE
    return (score - lo) / (hi - lo)


def denormalize_asap2(norm_score: float) -> float:
    """Convert [0, 1] back to ASAP2 scale (1-6)."""
    lo, hi = ASAP2_SCORE_RANGE
    return round(norm_score * (hi - lo) + lo)


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════

def load_asap1(path: str) -> pd.DataFrame:
    """
    Load ASAP1 labeled training data with averaged rater scores.
    Score = (rater1_domain1 + rater2_domain1) / 2
    """
    df = pd.read_csv(path, sep='\t', encoding='ISO-8859-1')

    result = pd.DataFrame()
    result['essay_id'] = df['essay_id']
    result['prompt_id'] = df['essay_set']
    result['raw_score'] = (df['rater1_domain1'] + df['rater2_domain1']) / 2.0

    result['text'] = df.apply(
        lambda row: build_text_input(
            row['essay'],
            ASAP1_PROMPT_DESCRIPTIONS.get(row['essay_set'], ""),
        ),
        axis=1,
    )

    result['norm_score'] = result.apply(
        lambda row: normalize_asap1(row['raw_score'], row['prompt_id']),
        axis=1,
    )

    return result


def load_asap2(path: str) -> pd.DataFrame:
    """Load ALL of ASAP2. Short prompt descriptions only (no source texts)."""
    df = pd.read_csv(path)

    unique_prompts = sorted(df['prompt_name'].unique())
    prompt_map = {p: i + 100 for i, p in enumerate(unique_prompts)}

    result = pd.DataFrame()
    result['essay_id'] = df['essay_id']
    result['prompt_name'] = df['prompt_name']
    result['prompt_id'] = df['prompt_name'].map(prompt_map)

    result['text'] = df.apply(
        lambda row: build_text_input(
            row['full_text'],
            ASAP2_PROMPT_DESCRIPTIONS.get(row['prompt_name'], ''),
        ),
        axis=1,
    )

    result['raw_score'] = df['score'].fillna(0).astype(float)
    result['norm_score'] = result['raw_score'].apply(normalize_asap2)

    return result


def split_train_val(
    df: pd.DataFrame,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ASAP1 into train/val, stratified by prompt_id."""
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(sss.split(df, df['prompt_id']))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def load_all_data(
    asap1_path: str = './data/training_set_rel3.tsv',
    asap2_path: str = './data/ASAP2_train_sourcetexts.csv',
    val_fraction: float = DEFAULT_VAL_FRACTION,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """
    Load and prepare ALL data for the ASAP1→ASAP2 transfer task.

    Returns a dict with:
        data['asap1_train']  – 11,029 labeled training essays
        data['asap1_val']    –  1,947 labeled validation essays
        data['asap2']        – 24,728 ASAP2 essays (held-out test)
        data['config']       – parameters used
    """
    print("=" * 60)
    print("Shared Data Preparation")
    print("=" * 60)

    # ASAP1
    print(f"\n[1/2] Loading ASAP1: {asap1_path}")
    asap1_all = load_asap1(asap1_path)
    asap1_train, asap1_val = split_train_val(asap1_all, val_fraction, seed)

    print(f"  Total: {len(asap1_all)} essays")
    print(f"  Train: {len(asap1_train)}, Val: {len(asap1_val)}")

    print(f"\n  Per-prompt summary:")
    for pid in sorted(asap1_all['prompt_id'].unique()):
        sub = asap1_all[asap1_all['prompt_id'] == pid]
        lo, hi = ASAP1_RATER_RANGES[pid]
        print(f"    Prompt {pid}: n={len(sub):>5}, "
              f"rater range [{lo}-{hi}], "
              f"avg [{sub['raw_score'].min():.1f}-{sub['raw_score'].max():.1f}], "
              f"norm [{sub['norm_score'].min():.3f}-{sub['norm_score'].max():.3f}]")

    # ASAP2
    print(f"\n[2/2] Loading ASAP2: {asap2_path}")
    asap2 = load_asap2(asap2_path)
    print(f"  Total: {len(asap2)} essays (ALL held out for evaluation)")
    print(f"  Score range: {int(asap2['raw_score'].min())}-{int(asap2['raw_score'].max())}")

    config = {
        'scoring': 'averaged raters (rater1 + rater2) / 2',
        'normalization': 'per-prompt min-max to [0, 1]',
        'split': f'{int((1-val_fraction)*100)}/{int(val_fraction*100)} stratified by prompt',
        'seed': seed,
        'n_train': len(asap1_train),
        'n_val': len(asap1_val),
        'n_asap2': len(asap2),
    }

    print(f"\n{'=' * 60}")
    print("Columns in every DataFrame:")
    print("  essay_id, prompt_id, text, raw_score, norm_score")
    print(f"{'=' * 60}")

    return {
        'asap1_train': asap1_train,
        'asap1_val': asap1_val,
        'asap2': asap2,
        'config': config,
    }


if __name__ == '__main__':
    data = load_all_data()

    print("\n-- Sample ASAP1 Train Row --")
    row = data['asap1_train'].iloc[0]
    print(f"  essay_id={row['essay_id']}, prompt={row['prompt_id']}, "
          f"raw={row['raw_score']}, norm={row['norm_score']:.4f}")
    print(f"  text: {row['text'][:120]}...")

    print("\n-- Sample ASAP2 Row --")
    row = data['asap2'].iloc[0]
    print(f"  essay_id={row['essay_id']}, prompt={row['prompt_id']}, "
          f"raw={row['raw_score']}, norm={row['norm_score']:.4f}")
    print(f"  text: {row['text'][:120]}...")