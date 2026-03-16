import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizerFast
from peft import LoraConfig, get_peft_model

from config import ModelConfig, LoRAConfig


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_

class CORNOrdinalHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes - 1)

    def forward(self, x):
        return self.fc(x)

    def predict_labels(self, logits):
        probs = torch.sigmoid(logits)
        cumulative = torch.cumprod(probs, dim=1)
        return (cumulative > 0.5).sum(dim=1)

    def predict_probs(self, logits):
        K = self.num_classes
        cond_probs = torch.sigmoid(logits)
        cum = torch.cumprod(cond_probs, dim=1)
        probs = torch.zeros(logits.size(0), K, device=logits.device)
        probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, K - 1):
            probs[:, k] = cum[:, k - 1] - cum[:, k]
        probs[:, K - 1] = cum[:, K - 2]
        return probs.clamp(min=1e-7)

    def predict_expected_score(self, logits, norm: bool = True):
        probs = torch.sigmoid(logits)
        cumulative = torch.cumprod(probs, dim=1)
        expected = cumulative.sum(dim=1)
        if norm:
            expected = expected / (self.num_classes - 1)
        return expected

class AESDAModel(nn.Module):

    def __init__(self, model_cfg: ModelConfig):
        super().__init__()
        self.model_cfg = model_cfg

        self.longformer = LongformerModel.from_pretrained(model_cfg.backbone)

        self.feature_proj = nn.Sequential(
            nn.Linear(model_cfg.hidden_dim, model_cfg.hidden_dim),
            nn.Tanh(),
            nn.Dropout(model_cfg.dropout),
        )

        self.regression_head = nn.Linear(model_cfg.hidden_dim, 1)
        self.ordinal_head = CORNOrdinalHead(model_cfg.hidden_dim, model_cfg.num_classes)

        self.grl = GradientReversalLayer(lambda_=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(model_cfg.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(256, 2),
        )

    def encode(self, input_ids, attention_mask, global_attention_mask=None):
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.feature_proj(cls_output)

    def forward(self, input_ids, attention_mask, global_attention_mask=None):
        features = self.encode(input_ids, attention_mask, global_attention_mask)
        reg_score = torch.sigmoid(self.regression_head(features).squeeze(-1))
        ordinal_logits = self.ordinal_head(features)
        reversed_features = self.grl(features)
        domain_logits = self.domain_head(reversed_features)
        return {
            'features': features,
            'reg_score': reg_score,
            'ordinal_logits': ordinal_logits,
            'domain_logits': domain_logits,
        }

    def predict(self, input_ids, attention_mask, global_attention_mask=None):
        features = self.encode(input_ids, attention_mask, global_attention_mask)
        reg_score = torch.sigmoid(self.regression_head(features).squeeze(-1))
        ordinal_logits = self.ordinal_head(features)
        ordinal_labels = self.ordinal_head.predict_labels(ordinal_logits)
        ordinal_probs = self.ordinal_head.predict_probs(ordinal_logits)
        ordinal_expected = self.ordinal_head.predict_expected_score(ordinal_logits, norm=True)
        return {
            'features': features,
            'reg_score': reg_score,
            'ordinal_logits': ordinal_logits,
            'ordinal_labels': ordinal_labels,
            'ordinal_probs': ordinal_probs,
            'ordinal_expected': ordinal_expected,
        }

    def set_dann_lambda(self, lambda_: float):
        self.grl.set_lambda(lambda_)

def apply_lora_da(model: AESDAModel, lora_cfg: LoRAConfig) -> AESDAModel:
    peft_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
    )
    model.longformer = get_peft_model(model.longformer, peft_config)
    model.longformer.print_trainable_parameters()
    return model


def get_tokenizer(model_cfg: ModelConfig) -> LongformerTokenizerFast:
    tokenizer = LongformerTokenizerFast.from_pretrained(model_cfg.backbone)
    if "<PARA>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<PARA>"])
    return tokenizer


def resize_embeddings(model: AESDAModel, tokenizer: LongformerTokenizerFast):
    model.longformer.resize_token_embeddings(len(tokenizer))


def load_stage_s_into_da(
    model: AESDAModel,
    stage_s_path: str,
    device: torch.device,
) -> AESDAModel:
    state_dict = torch.load(stage_s_path, map_location=device)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    result = model.load_state_dict(state_dict, strict=False)

    print(f"  Loaded Stage S checkpoint: {stage_s_path}")
    print(f"  Matched keys: {len(ckpt_keys & model_keys)}")
    if result.missing_keys:
        da_missing = [k for k in result.missing_keys
                      if 'domain_head' in k or 'grl' in k]
        other_missing = [k for k in result.missing_keys
                         if 'domain_head' not in k and 'grl' not in k]
        print(f"  DA-specific (randomly initialized): {len(da_missing)} keys")
        if other_missing:
            print(f"  WARNING – other missing keys: {other_missing[:5]}")
    if result.unexpected_keys:
        print(f"  Unexpected keys (ignored): {len(result.unexpected_keys)}")

    return model
