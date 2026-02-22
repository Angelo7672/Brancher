import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# ------------------------
# Loss functions for testing
# ------------------------

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)


def per_jpe(predicted, target):
    """
    per-joint position error
    """
    assert predicted.shape == target.shape
    return np.linalg.norm(predicted - target, axis=len(target.shape) - 1)


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)


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


# Bone length loss
def get_limb_lens(x):
    '''
        Input: (N, T, 17, 3)
        Output: (N, T, 18)
    '''
    limbs_id = [  [15, 13], [13, 11], [16, 14],
                  [14, 12], [11, 12], [5, 11], [6, 12],
                  [5, 6], [5, 7], [6, 8], [7, 9],
                  [8, 10], [5, 3], [6, 4], [3, 1],
                  [4, 2], [1, 0], [2, 0] ]       
    limbs = x[:,:,limbs_id,:]
    limbs = limbs[:,:,:,0,:]-limbs[:,:,:,1,:]   # vettore osso
    limb_lens = torch.norm(limbs, dim=-1)
    return limb_lens

def loss_limb_var(x):
    '''
        Input: (N, T, 17, 3)
    '''
    if x.shape[1]<=1:
        return torch.FloatTensor(1).fill_(0.)[0].to(x.device)
    limb_lens = get_limb_lens(x)
    limb_lens_var = torch.var(limb_lens, dim=1)
    limb_loss_var = torch.mean(limb_lens_var)
    return limb_loss_var


# Velocity smoothness loss
def velocity_smoothness_loss(x, power=2):
    """
    Penalizza variazioni improvvise di velocità (movimento non fluido)
    Input: x -> (N, T, J, 3)
    """
    if x.shape[1] < 3:
        return torch.zeros(1, device=x.device)

    # velocità
    v = x[:, 1:] - x[:, :-1]         # (N, T-1, J, 3)
    # variazione della velocità nel tempo (accelerazione implicita)
    dv = v[:, 1:] - v[:, :-1]        # (N, T-2, J, 3)

    if power == 2:
        # Norma L2 al quadrato: molto efficace contro i cambiamenti bruschi
        loss = torch.mean(dv**2)
    else:
        # Norma L1: più tollerante verso piccoli movimenti naturali
        loss = torch.mean(torch.norm(dv, dim=-1))

    return loss


# Acceleration smoothness loss
def acceleration_smoothness_loss(x, power=2):
    """
    Penalizza accelerazioni troppo brusche (jerk fisicamente innaturale)
    Input: x -> (N, T, J, 3)
    """
    if x.shape[1] < 4:
        return torch.zeros(1, device=x.device)

    # accelerazione
    a = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]  # (N, T-2, J, 3)
    # variazione dell'accelerazione (jerk)
    da = a[:, 1:] - a[:, :-1]                  # (N, T-3, J, 3)
    
    if power == 2:
        # Norma L2 al quadrato: molto efficace contro i cambiamenti bruschi
        loss = torch.mean(da**2)
    else:
        # Norma L1: più tollerante verso piccoli movimenti naturali
        loss = torch.mean(torch.norm(da, dim=-1))

    return loss


# Supervised velocity loss
def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))