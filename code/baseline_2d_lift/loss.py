import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# ------------------------
# Loss functions
# ------------------------

# Mean Per Joint Position Error
def loss_mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))


# Mean Per Joint Position Error after Procrustes Alignment
def loss_p_mpjpe(predicted, target):
    """
    Protocol #2 MPJPE (rigid alignment) per batch di sequenze (B,T,J,3)
    """
    assert predicted.shape == target.shape, \
        f"Shape mismatch: {predicted.shape} vs {target.shape}"

    B, T, J, D = predicted.shape
    predicted = predicted.reshape(B*T, J, D)
    target = target.reshape(B*T, J, D)

    muX = target.mean(1, keepdim=True)
    muY = predicted.mean(1, keepdim=True)

    X0 = target - muX
    Y0 = predicted - muY

    X0 /= X0.norm(dim=(1,2), keepdim=True)
    Y0 /= Y0.norm(dim=(1,2), keepdim=True)

    H = X0.transpose(1,2) @ Y0
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1,2)
    R = V @ U.transpose(1,2)

    detR = torch.linalg.det(R)
    neg_mask = detR < 0
    if neg_mask.any():
        V[neg_mask, :, -1] *= -1
        R[neg_mask] = V[neg_mask] @ U[neg_mask].transpose(1,2)

    scale = (s.sum(1, keepdim=True).unsqueeze(2) *
             X0.norm(dim=(1,2), keepdim=True) /
             Y0.norm(dim=(1,2), keepdim=True))

    t = muX - scale * (muY @ R)
    aligned = scale * (predicted @ R) + t

    return torch.mean(torch.norm(aligned - target, dim=-1))