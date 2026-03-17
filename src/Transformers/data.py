import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast
from typing import List, Optional, Tuple
import re
import json

from config import ASAP1_RATER_RANGES, ASAP2_SCORE_RANGE, Paths, ModelConfig

def normalize_asap1_avg(avg_score: float, prompt_id: int) -> float:
    lo, hi = ASAP1_RATER_RANGES[prompt_id]
    if hi == lo:
        return 0.5
    return (avg_score - lo) / (hi - lo)


def denormalize_asap1_avg(norm_score: float, prompt_id: int) -> float:
    lo, hi = ASAP1_RATER_RANGES[prompt_id]
    return norm_score * (hi - lo) + lo


def normalize_asap2_score(score: float) -> float:
    lo, hi = ASAP2_SCORE_RANGE
    return (score - lo) / (hi - lo)


def denormalize_asap2_score(norm_score: float) -> float:
    lo, hi = ASAP2_SCORE_RANGE
    return norm_score * (hi - lo) + lo

def compute_quantile_bins(
    norm_scores: np.ndarray, num_bins: int = 6
) -> np.ndarray:
    percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    bin_edges = np.percentile(norm_scores, percentiles)

    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < len(bin_edges):
        unique_edges = np.linspace(
            norm_scores.min() + 1e-6, norm_scores.max() - 1e-6, num_bins - 1
        )

    return unique_edges


def assign_ordinal_label(
    norm_score: float, bin_edges: np.ndarray
) -> int:
    return int(np.digitize(norm_score, bin_edges))


def save_bin_edges(bin_edges: np.ndarray, path: str):
    with open(path, 'w') as f:
        json.dump({'bin_edges': bin_edges.tolist(), 'num_bins': len(bin_edges) + 1}, f)


def load_bin_edges(path: str) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    return np.array(data['bin_edges'])

def preprocess_essay(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{2,}', ' <PARA> ', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def build_prompt_essay_input(essay_text: str, prompt_text: str = "",
                             source_snippet: str = "") -> str:
    essay_text = preprocess_essay(essay_text)
    parts = []
    if prompt_text:
        parts.append(preprocess_essay(prompt_text))
    if source_snippet:
        parts.append(preprocess_essay(source_snippet))
    parts.append(essay_text)
    return " </s> ".join(parts)


def truncate_source_texts(source_texts: List[str], max_words: int = 300) -> str:
    combined = []
    for st in source_texts:
        if isinstance(st, str) and st.strip() and st.strip().upper() != "NA":
            combined.append(st.strip())
    if not combined:
        return ""
    full_text = " ".join(combined)
    words = full_text.split()
    return " ".join(words[:max_words])


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

class AESDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        norm_scores: List[float],
        ordinal_labels: List[int],
        domain_labels: List[int],
        prompt_ids: List[int],
        raw_scores: List[float],
        has_labels: List[bool],
        tokenizer: LongformerTokenizerFast,
        max_length: int = 2048,
    ):
        self.texts = texts
        self.norm_scores = norm_scores
        self.ordinal_labels = ordinal_labels
        self.domain_labels = domain_labels
        self.prompt_ids = prompt_ids
        self.raw_scores = raw_scores
        self.has_labels = has_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[0] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "global_attention_mask": global_attention_mask,
            "norm_score": torch.tensor(self.norm_scores[idx], dtype=torch.float32),
            "ordinal_label": torch.tensor(self.ordinal_labels[idx], dtype=torch.long),
            "domain_label": torch.tensor(self.domain_labels[idx], dtype=torch.long),
            "prompt_id": torch.tensor(self.prompt_ids[idx], dtype=torch.long),
            "raw_score": torch.tensor(self.raw_scores[idx], dtype=torch.float32),
            "has_label": torch.tensor(self.has_labels[idx], dtype=torch.bool),
        }

def load_asap1_labeled(
    paths: Paths,
    bin_edges: Optional[np.ndarray] = None,
    num_classes: int = 6,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(paths.asap1_train_tsv, sep='\t', encoding='ISO-8859-1')

    result = pd.DataFrame()
    result['essay_id'] = df['essay_id']
    result['prompt_id'] = df['essay_set']

    result['raw_score'] = (df['rater1_domain1'] + df['rater2_domain1']) / 2.0

    for pid in sorted(result['prompt_id'].unique()):
        lo, hi = ASAP1_RATER_RANGES[pid]
        subset = result[result['prompt_id'] == pid]
        avg_min, avg_max = subset['raw_score'].min(), subset['raw_score'].max()
        assert avg_min >= lo, f"Prompt {pid}: avg min {avg_min} < range lo {lo}"
        assert avg_max <= hi, f"Prompt {pid}: avg max {avg_max} > range hi {hi}"

    result['text'] = df.apply(
        lambda row: build_prompt_essay_input(
            row['essay'],
            ASAP1_PROMPT_DESCRIPTIONS.get(row['essay_set'], "")
        ),
        axis=1,
    )

    result['norm_score'] = result.apply(
        lambda row: normalize_asap1_avg(row['raw_score'], row['prompt_id']),
        axis=1,
    )

    if bin_edges is None:
        bin_edges = compute_quantile_bins(result['norm_score'].values, num_classes)
        print(f"  Computed quantile bin edges ({num_classes} bins): "
              f"{np.round(bin_edges, 4)}")

    result['ordinal_label'] = result['norm_score'].apply(
        lambda s: assign_ordinal_label(s, bin_edges)
    )
    result['domain_label'] = 0
    result['has_label'] = True

    print(f"  Ordinal label distribution:")
    for k in range(num_classes):
        n = (result['ordinal_label'] == k).sum()
        pct = 100 * n / len(result)
        print(f"    Bin {k}: {n:>5} ({pct:.1f}%)")

    return result, bin_edges


def split_asap1_train_val(
    df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss.split(df, df['prompt_id']))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def _load_asap1_unlabeled_tsv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', encoding='ISO-8859-1')
    result = pd.DataFrame()
    result['essay_id'] = df['essay_id']
    result['prompt_id'] = df['essay_set']
    result['text'] = df.apply(
        lambda row: build_prompt_essay_input(
            row['essay'],
            ASAP1_PROMPT_DESCRIPTIONS.get(row['essay_set'], "")
        ),
        axis=1,
    )
    result['raw_score'] = 0.0
    result['norm_score'] = 0.0
    result['ordinal_label'] = 0
    result['domain_label'] = 0
    result['has_label'] = False
    return result

ASAP2_PROMPT_DESCRIPTIONS = {
    '"A Cowboy Who Rode the Waves"': "Source-based: describe the narrator's perspective and experiences.",
    'Car-free cities': "Write about the advantages or disadvantages of limiting car usage in cities.",
    'Does the electoral college work?': "Write about whether the electoral college system is effective.",
    'Driverless cars': "Write about the development and implications of driverless car technology.",
    'Exploring Venus': "Write about the challenges and value of exploring the planet Venus.",
    'Facial action coding system': "Write about the value of technology that reads facial expressions.",
    'The Face on Mars': "Write about the landform on Mars known as the Face and what it represents.",
}


def load_asap2(
    paths: Paths,
    bin_edges: np.ndarray,
    num_classes: int = 6,
    source_max_words: int = 300,
    strip_source: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(paths.asap2_train_csv)

    unique_prompts = sorted(df['prompt_name'].unique())
    prompt_map = {p: i + 100 for i, p in enumerate(unique_prompts)}
    print(f"  ASAP2 prompts: {list(prompt_map.keys())}")

    result = pd.DataFrame()
    result['essay_id'] = df['essay_id']
    result['prompt_name'] = df['prompt_name']
    result['prompt_id'] = df['prompt_name'].map(prompt_map)

    if strip_source:
        print("  Input format: SHORT prompt + essay (matching ASAP1 structure)")
        result['text'] = df.apply(
            lambda row: build_prompt_essay_input(
                essay_text=row['full_text'],
                prompt_text=ASAP2_PROMPT_DESCRIPTIONS.get(row['prompt_name'], ''),
            ),
            axis=1,
        )
    else:
        source_cols = sorted([c for c in df.columns if c.startswith('source_text_')])
        result['text'] = df.apply(
            lambda row: build_prompt_essay_input(
                essay_text=row['full_text'],
                prompt_text=row.get('assignment', ''),
                source_snippet=truncate_source_texts(
                    [row[c] for c in source_cols], max_words=source_max_words
                ),
            ),
            axis=1,
        )

    result['raw_score'] = df['score'].fillna(0).astype(float)
    result['norm_score'] = result['raw_score'].apply(normalize_asap2_score)
    result['ordinal_label'] = result['norm_score'].apply(
        lambda s: assign_ordinal_label(s, bin_edges)
    )
    result['has_label'] = ~df['score'].isna()
    result['domain_label'] = 1

    return result

def build_dataset(df: pd.DataFrame, tokenizer, max_length: int = 2048) -> AESDataset:
    return AESDataset(
        texts=df['text'].tolist(),
        norm_scores=df['norm_score'].tolist(),
        ordinal_labels=df['ordinal_label'].tolist(),
        domain_labels=df['domain_label'].tolist(),
        prompt_ids=df['prompt_id'].tolist(),
        raw_scores=df['raw_score'].tolist(),
        has_labels=df['has_label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
