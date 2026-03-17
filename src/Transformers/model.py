import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerTokenizerFast
from peft import LoraConfig, get_peft_model

from config import ModelConfig, LoRAConfig

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
        predicted = (cumulative > 0.5).sum(dim=1)
        return predicted

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

class AESTransferModel(nn.Module):

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
        return {
            'features': features,
            'reg_score': reg_score,
            'ordinal_logits': ordinal_logits,
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

def apply_lora(model: AESTransferModel, lora_cfg: LoRAConfig) -> AESTransferModel:
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


def freeze_backbone_except_top(model: AESTransferModel, n_layers: int = 2):
    for param in model.longformer.parameters():
        param.requires_grad = False

    total_layers = len(model.longformer.encoder.layer)
    for i in range(total_layers - n_layers, total_layers):
        for param in model.longformer.encoder.layer[i].parameters():
            param.requires_grad = True

    if hasattr(model.longformer, 'pooler') and model.longformer.pooler is not None:
        for param in model.longformer.pooler.parameters():
            param.requires_grad = True

    for module in [model.feature_proj, model.regression_head, model.ordinal_head]:
        for param in module.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Two-Stage S1: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")


def unfreeze_all(model: AESTransferModel):
    for param in model.parameters():
        param.requires_grad = True


def get_tokenizer(model_cfg: ModelConfig) -> LongformerTokenizerFast:
    tokenizer = LongformerTokenizerFast.from_pretrained(model_cfg.backbone)
    if "<PARA>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<PARA>"])
    return tokenizer


def resize_embeddings(model: AESTransferModel, tokenizer: LongformerTokenizerFast):
    model.longformer.resize_token_embeddings(len(tokenizer))
