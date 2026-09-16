"""Training objective from manuscript_0811_polished, Eqs. (16)-(21)."""

import torch
from torch import nn
from torch.nn import functional as F


def orthogonality_loss(shared, private, mask=None):
    """Mean_b ||S_b.T P_b / L_b||_F^2 over valid tokens, Eq. (18).

    For short sequences the equivalent token-Gram contraction avoids a
    [B, d, d] intermediate: ||S.T P||_F^2 = sum((S S.T) * (P P.T)).
    No feature normalization, token sampling or division by d is applied.
    """
    if shared.ndim != 3 or shared.shape != private.shape or min(shared.shape) < 1:
        raise ValueError("Shared and private features must have equal nonempty [B, L, d] shapes.")
    if mask is not None:
        if mask.dtype != torch.bool or mask.shape != shared.shape[:2] or mask.device != shared.device:
            raise ValueError("mask must be a boolean [B, L] padding mask on the feature device.")
        lengths = (~mask).sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("Orthogonality requires at least one valid token per sample.")
        shared = shared.masked_fill(mask.unsqueeze(-1), 0.0)
        private = private.masked_fill(mask.unsqueeze(-1), 0.0)
    else:
        lengths = shared.new_full((shared.shape[0],), shared.shape[1])

    # Squared cross-correlations can overflow fp16 under mixed precision.
    with torch.autocast(device_type=shared.device.type, enabled=False):
        if shared.dtype in (torch.float16, torch.bfloat16):
            shared, private = shared.float(), private.float()
        if shared.shape[1] < shared.shape[2]:
            gram_s = torch.bmm(shared, shared.transpose(1, 2))
            gram_p = torch.bmm(private, private.transpose(1, 2))
            squared_norm = (gram_s * gram_p).sum(dim=(1, 2)).clamp_min(0.0)
        else:
            cross = torch.bmm(shared.transpose(1, 2), private)
            squared_norm = cross.square().sum(dim=(1, 2))
        return (squared_norm / lengths.to(shared.dtype).square()).mean()


class KCNetObjective(nn.Module):
    """Eq. (21), summing experts/pairs rather than averaging their losses.

    The manuscript does not report selected alpha/lambda values. These
    configurable defaults retain the release's 0.3 weights and 1e-4 penalty
    coefficient; tune them on validation data for the new objective.
    """

    def __init__(self, alpha_aux=0.3, alpha_orth=0.3, alpha_con=0.3, lambda_sp=1e-4):
        super().__init__()
        weights = (alpha_aux, alpha_orth, alpha_con, lambda_sp)
        if any(weight < 0 for weight in weights):
            raise ValueError("Loss weights must be nonnegative.")
        self.alpha_aux, self.alpha_orth, self.alpha_con, self.lambda_sp = weights

    def forward(self, model, outputs, labels, masks=None):
        logits, expert_logits, _, shared, private, shared_pools = outputs
        count = len(model.expert_names)
        if not all(len(items) == count for items in (expert_logits, shared, private, shared_pools)):
            raise ValueError("Outputs must follow the model's expert order.")
        masks = [None] * count if masks is None else list(masks)
        if len(masks) != count:
            raise ValueError("Provide one padding mask per expert.")

        main = F.cross_entropy(logits, labels)
        auxiliary = torch.stack([F.cross_entropy(prediction, labels) for prediction in expert_logits]).sum()
        orthogonal = torch.stack([
            orthogonality_loss(s, p, mask) for s, p, mask in zip(shared, private, masks)
        ]).sum()
        pools = dict(zip(model.expert_names, shared_pools))
        pair_losses = [F.mse_loss(pools[i], pools[j]) for i, j in model.consistency_pairs]
        consistency = torch.stack(pair_losses).sum() if pair_losses else main.new_zeros(())
        sparsity = self.lambda_sp * model.nbc_sparsity_loss()
        total = (main + self.alpha_aux * auxiliary + self.alpha_orth * orthogonal
                 + self.alpha_con * consistency + sparsity)
        return {"total": total, "main": main, "aux": auxiliary, "orth": orthogonal,
                "con": consistency, "sparsity": sparsity}
