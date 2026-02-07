import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# ------------------------
# Loss functions
# ------------------------

# Mean Per Joint Position Error con Masking Outlier
def loss_mpjpe(predicted, target, uav=False, threshold=2.0):
    """
    Se uav=True, ignora i primi 5 joint.
    Se un joint nel target supera threshold (metri dalla root), viene ignorato.
    """
    assert predicted.shape == target.shape
    
    if uav:
        predicted = predicted[:, :, 5:, :]
        target = target[:, :, 5:, :]
        
    # Calcolo errore euclideo per ogni joint
    dist_err = torch.norm(predicted - target, dim=-1) # (B, T, J)
    
    # Calcolo distanza del target dalla root (già centrato in 0,0,0)
    target_dist_from_origin = torch.norm(target, dim=-1)
    
    # Maschera: 1 se il dato è valido, 0 se è un outlier
    valid_mask = (target_dist_from_origin <= threshold).float()
    
    # Applichiamo la maschera
    masked_err = dist_err * valid_mask
    
    # Media pesata solo sugli elementi validi
    return masked_err.sum() / (valid_mask.sum() + 1e-6)


# Mean Per Joint Position Error after Procrustes Alignment con Masking Outlier
def loss_p_mpjpe(predicted, target, uav=False, threshold=2.0):
    """
    Protocol #2 con filtraggio outlier.
    L'allineamento di Procruste viene calcolato solo sui joint validi.
    """
    assert predicted.shape == target.shape

    if uav:
        predicted = predicted[:, :, 5:, :]
        target = target[:, :, 5:, :]

    B, T, J, D = predicted.shape
    
    # Calcoliamo la maschera di validità prima del reshape
    # (B, T, J) -> (B*T, J, 1) per broadcasting
    target_dist = torch.norm(target, dim=-1)
    valid_mask = (target_dist <= threshold).float().reshape(B*T, J, 1)
    
    predicted = predicted.reshape(B*T, J, D)
    target = target.reshape(B*T, J, D)

    # Nota: Per Procrustes l'ideale sarebbe pesare le medie per la maschera
    # ma per semplicità e stabilità, filtriamo i contributi degli outlier
    muX = (target * valid_mask).sum(1, keepdim=True) / (valid_mask.sum(1, keepdim=True) + 1e-6)
    muY = (predicted * valid_mask).sum(1, keepdim=True) / (valid_mask.sum(1, keepdim=True) + 1e-6)

    X0 = (target - muX) * valid_mask
    Y0 = (predicted - muY) * valid_mask

    # ... [Resto del calcolo SVD rimane invariato, lavorando su X0 e Y0 già mascherati] ...
    X0_norm = X0.norm(dim=(1,2), keepdim=True) + 1e-7
    Y0_norm = Y0.norm(dim=(1,2), keepdim=True) + 1e-7
    X0 /= X0_norm
    Y0 /= Y0_norm

    H = X0.transpose(1,2) @ Y0
    U, s, Vt = torch.linalg.svd(H)
    V = Vt.transpose(1,2)
    R = V @ U.transpose(1,2)

    detR = torch.linalg.det(R)
    neg_mask = (detR < 0)
    if neg_mask.any():
        V[neg_mask, :, -1] *= -1
        R[neg_mask] = V[neg_mask] @ U[neg_mask].transpose(1,2)

    scale = (s.sum(1, keepdim=True).unsqueeze(2) * X0_norm / Y0_norm)
    t = muX - scale * (muY @ R)
    aligned = scale * (predicted @ R) + t

    # Calcolo errore finale mascherato
    final_err = torch.norm(aligned - target, dim=-1) * valid_mask.squeeze(-1)
    return final_err.sum() / (valid_mask.sum() + 1e-6)