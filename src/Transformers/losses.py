import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

try:
    from coral_pytorch.losses import corn_loss as _coral_corn_loss
    HAS_CORAL_PYTORCH = True
except ImportError:
    HAS_CORAL_PYTORCH = False

class SoftQWKLoss(nn.Module):

    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.num_classes = num_classes
        K = num_classes
        w = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                w[i, j] = ((i - j) ** 2) / ((K - 1) ** 2)
        self.register_buffer('weight_matrix', w)

    def forward(
        self,
        ordinal_logits: torch.Tensor,
        true_labels: torch.Tensor,
    ) -> torch.Tensor:
        K = self.num_classes
        B = ordinal_logits.size(0)

        if B < 2:
            return torch.tensor(0.0, device=ordinal_logits.device, requires_grad=True)

        cond_probs = torch.sigmoid(ordinal_logits)

        cum = torch.cumprod(cond_probs, dim=1)

        pred_probs = torch.zeros(B, K, device=ordinal_logits.device)
        pred_probs[:, 0] = 1.0 - cum[:, 0]
        for k in range(1, K - 1):
            pred_probs[:, k] = cum[:, k - 1] - cum[:, k]
        pred_probs[:, K - 1] = cum[:, K - 2]

        pred_probs = pred_probs.clamp(min=1e-7)
        pred_probs = pred_probs / pred_probs.sum(dim=1, keepdim=True)

        true_onehot = F.one_hot(true_labels, K).float()

        O = true_onehot.T @ pred_probs
        O = O / O.sum().clamp(min=1e-7)

        hist_true = true_onehot.sum(dim=0)
        hist_pred = pred_probs.sum(dim=0)
        E = hist_true.unsqueeze(1) * hist_pred.unsqueeze(0)
        E = E / E.sum().clamp(min=1e-7)

        W = self.weight_matrix.to(O.device)
        num = (W * O).sum()
        den = (W * E).sum()

        if den < 1e-7:
            return torch.tensor(0.0, device=ordinal_logits.device, requires_grad=True)

        soft_qwk = 1.0 - num / den
        loss = 1.0 - soft_qwk

        return loss

def corn_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if HAS_CORAL_PYTORCH:
        return _coral_corn_loss(logits, labels, num_classes=num_classes)

    total_loss = torch.tensor(0.0, device=logits.device)
    count = 0
    for k in range(num_classes - 1):
        mask = labels >= k
        if mask.sum() == 0:
            continue
        target = (labels[mask] > k).float()
        loss_k = F.binary_cross_entropy_with_logits(logits[mask, k], target)
        total_loss = total_loss + loss_k
        count += 1
    return total_loss / max(count, 1)

class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 0.5, max_pairs: int = 512):
        super().__init__()
        self.margin = margin
        self.max_pairs = max_pairs

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pred_diff = predictions.unsqueeze(1) - predictions.unsqueeze(0)
        target_sign = torch.sign(targets.unsqueeze(1) - targets.unsqueeze(0))

        mask = torch.triu(torch.ones_like(target_sign, dtype=torch.bool), diagonal=1)
        mask = mask & (target_sign != 0)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        if mask.sum() > self.max_pairs:
            indices = torch.nonzero(mask, as_tuple=False)
            perm = torch.randperm(len(indices))[:self.max_pairs]
            selected = indices[perm]
            pred_d = pred_diff[selected[:, 0], selected[:, 1]]
            tgt_s = target_sign[selected[:, 0], selected[:, 1]]
        else:
            pred_d = pred_diff[mask]
            tgt_s = target_sign[mask]

        hinge = torch.clamp(self.margin - tgt_s * pred_d, min=0.0)
        return hinge.mean()

class AESMultiTaskLoss(nn.Module):

    def __init__(
        self,
        num_classes: int = 6,
        mse_weight: float = 1.0,
        ranking_weight: float = 0.5,
        ordinal_weight: float = 0.5,
        soft_qwk_weight: float = 1.0,
        ranking_margin: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.mse_weight = mse_weight
        self.ranking_weight = ranking_weight
        self.ordinal_weight = ordinal_weight
        self.soft_qwk_weight = soft_qwk_weight

        self.mse_loss = nn.MSELoss()
        self.ranking_loss = PairwiseRankingLoss(margin=ranking_margin)
        self.soft_qwk_loss = SoftQWKLoss(num_classes=num_classes)

    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        norm_scores: torch.Tensor,
        ordinal_labels: torch.Tensor,
        has_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        dev = norm_scores.device

        labeled_mask = has_label.bool()
        if labeled_mask.sum() > 0:
            reg = model_output['reg_score'][labeled_mask]
            targets = norm_scores[labeled_mask]
            ord_logits = model_output['ordinal_logits'][labeled_mask]
            ord_labels = ordinal_labels[labeled_mask]

            losses['mse'] = self.mse_loss(reg, targets) * self.mse_weight

            if labeled_mask.sum() >= 2:
                losses['ranking'] = self.ranking_loss(reg, targets) * self.ranking_weight
            else:
                losses['ranking'] = torch.tensor(0.0, device=dev)

            losses['ordinal'] = corn_loss(ord_logits, ord_labels, self.num_classes) * self.ordinal_weight

            losses['soft_qwk'] = self.soft_qwk_loss(ord_logits, ord_labels) * self.soft_qwk_weight
        else:
            losses['mse'] = torch.tensor(0.0, device=dev)
            losses['ranking'] = torch.tensor(0.0, device=dev)
            losses['ordinal'] = torch.tensor(0.0, device=dev)
            losses['soft_qwk'] = torch.tensor(0.0, device=dev)

        losses['total'] = sum(losses.values())
        return losses
