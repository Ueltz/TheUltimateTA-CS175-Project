from dataclasses import dataclass, field
import torch


@dataclass
class Paths:
    asap1_train_tsv: str = "./data/training_set_rel3.tsv"
    asap1_unlabeled_valid_tsv: str = "./data/valid_set.tsv"
    asap1_unlabeled_test_tsv: str = "./data/test_set.tsv"
    asap2_train_csv: str = "./data/ASAP2_train_sourcetexts.csv"

    output_dir: str = "./outputs"
    stage_s_dir: str = "./outputs/stage_s"
    stage_c_dir: str = "./outputs/stage_c"
    logs_dir: str = "./outputs/logs"
    graphs_dir: str = "./outputs/graphs"


@dataclass
class ModelConfig:
    backbone: str = "allenai/longformer-base-4096"
    max_length: int = 2048
    num_classes: int = 6
    hidden_dim: int = 768
    dropout: float = 0.1
    ordinal_head: bool = True
    regression_head: bool = True


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    target_modules: list = field(default_factory=lambda: ["query", "key", "value"])
    lora_dropout: float = 0.1
    bias: str = "none"
    two_stage: bool = True
    stage1_unfreeze_layers: int = 2


@dataclass
class TrainConfig:
    stage_s_epochs: int = 5
    stage_s_lr: float = 2e-5
    stage_s_lora_lr: float = 3e-4
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    fp16: bool = True
    gradient_checkpointing: bool = True

    mse_weight: float = 1.0
    ranking_weight: float = 0.5
    ordinal_weight: float = 0.5
    soft_qwk_weight: float = 1.0
    ranking_margin: float = 0.5

    stage_u_epochs: int = 5
    dann_weight: float = 1.0
    coral_weight: float = 1.0
    dann_lambda_schedule: bool = True

    ust_iterations: int = 3
    ust_mc_samples: int = 10
    ust_mc_dropout: float = 0.2
    ust_percentile: float = 25.0
    ust_lr: float = 2e-4

    seed: int = 42
    num_workers: int = 4
    eval_steps: int = 200
    patience: int = 3
    log_every_n_steps: int = 50

ASAP1_RATER_RANGES = {
    1: (1, 6),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (0, 12),
    8: (0, 30),
}

ASAP2_SCORE_RANGE = (1, 6)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
