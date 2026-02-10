import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# ------------------------
# Loss functions
# ------------------------

# Mean Per Joint Position Error
def loss_mpjpe(predicted, target, uav=False, threshold=2.0):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
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


# Mean Per Joint Position Error after Procrustes Alignment
def loss_p_mpjpe(predicted, target, uav=False, threshold=2.0):
    """
    Protocol #2 MPJPE (rigid alignment) per batch di sequenze (B,T,J,3)
    """
    assert predicted.shape == target.shape, \
        f"Shape mismatch: {predicted.shape} vs {target.shape}"

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


def geodesic_loss(pred_rot, target_rot):
    """
    Calcola la Geodesic Loss tra due matrici di rotazione.
    Formula: theta = acos((tr(R_pred @ R_target^T) - 1) / 2)
    
    pred_rot, target_rot: [..., 3, 3] (tensori di matrici di rotazione)
    """
    # 1. Calcoliamo la matrice di "differenza" R_diff = R1 * R2^T
    # [..., 3, 3] @ [..., 3, 3] -> [..., 3, 3]
    R_diff = torch.matmul(pred_rot, target_rot.transpose(-1, -2))
    
    # 2. Calcoliamo la traccia della matrice risultante
    # La traccia è la somma degli elementi sulla diagonale
    tr = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    
    # 3. Applichiamo la formula per l'angolo (in radianti)
    # Clamp indispensabile: a causa di errori di precisione floating point, 
    # l'argomento di acos potrebbe finire a 1.0000001 o -1.0000001, causando NaN.
    argument = (tr - 1) / 2.0
    argument = torch.clamp(argument, -1.0 + 1e-7, 1.0 - 1e-7)
    
    loss = torch.acos(argument)

    return loss.mean()


# Uncertainty Heatmap Loss 
def heatmap_loss(pred, gt):
    """
    pred, gt: (B, T, J, H, W) torch.Tensor
    return:   scalar torch.Tensor
    """
    return torch.mean((pred - gt) ** 2)


# ------------------------
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # self.log_vars rappresenta 's' nella formula: s = log(sigma^2).
        # Inizializzato a 0, il che significa sigma^2 = 1, e il peso iniziale exp(-0) = 1.
        # È un nn.Parameter, quindi l'ottimizzatore lo aggiornerà durante il backprop.
        #self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.log_vars = nn.Parameter(torch.zeros(5))

    def forward(self, loss_dict):
        """
        Riceve un dizionario dove ogni chiave è un task e ogni valore è la loss scalare (L_i).
        """
        total_loss = 0
        # L'ordine delle chiavi deve corrispondere all'indice in self.log_vars
        keys = ['mpjpe', 'bone', 'velocity', 'acceleration']#, 'geodesic', 'hm']
        
        for i, key in enumerate(keys):
            # Calcolo della precisione adattiva: exp(-s_i) = 1 / sigma_i^2
            precision = torch.exp(-self.log_vars[i])
            
            # Applicazione della formula di pesatura:
            # diff = (L_i / sigma_i^2) + log(sigma_i^2)
            # Nota: il termine log(sigma_i^2) funge da regolarizzatore per evitare che la rete aumenti sigma all'infinito per annullare la loss.
            # Nota: la formula matematica per la regressione include 0.5 davanti, ma ciò che conta è il bilanciamento relativo
            diff = precision * loss_dict[key] + self.log_vars[i]
            
            total_loss += diff
            
        return total_loss