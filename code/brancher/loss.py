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
def velocity_smoothness_loss(x):
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

    loss = torch.mean(dv**2)

    return loss


# Acceleration smoothness loss
def acceleration_smoothness_loss(x):
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

    loss = torch.mean(da**2)

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

def aleatoric_heatmap_loss(mu, log_var, target):
    """
    Calcola la loss con incertezza aleatoria per pixel.
    
    Args:
        mu: Heatmap predetta [B, T, J, H, W]
        log_var: Log-varianza predetta [B, T, J, H, W] (s nella formula)
        target: Heatmap di riferimento (GT) [B, T, J, H, W]
    """
    # 1. Calcoliamo la MSE per ogni joint (riduciamo solo H e W)
    # mse_per_joint shape: [B, T, J]
    mse_per_joint = torch.mean((mu - target)**2, dim=[-2, -1])
    
    # 2. Applichiamo la formula aleatoria standard
    # precision shape: [B, T, J]
    precision = torch.exp(-log_var)
    loss_per_joint = 0.5 * (precision * mse_per_joint + log_var)
    
    # 3. Media su tutto
    return torch.mean(loss_per_joint)


# ------------------------
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        # self.log_vars rappresenta 's' nella formula: s = log(sigma^2).
        # Inizializzato a 0, il che significa sigma^2 = 1, e il peso iniziale exp(-0) = 1.
        # È un nn.Parameter, quindi l'ottimizzatore lo aggiornerà durante il backprop.
        #self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.log_vars = nn.Parameter(torch.zeros(5))

        # TRUCCO: Se sai che 'hm' (indice 4) è molto piccola, 
        # puoi inizializzare il suo log_var a un valore negativo (es. -5)
        # così exp(-log_var) diventa un moltiplicatore grande (es. 148.4)
        with torch.no_grad():
            self.log_vars[4] = -5.0 # Peso iniziale maggiore per Heatmaps

    def forward(self, loss_dict):
        """
        Riceve un dizionario dove ogni chiave è un task e ogni valore è la loss scalare (L_i).
        """
        total_loss = 0
        # L'ordine delle chiavi deve corrispondere all'indice in self.log_vars
        keys = ['mpjpe', 'bone', 'velocity', 'acceleration', 'hm']#, 'geodesic', 'hm']
        
        for i, key in enumerate(keys):
            # Calcolo della precisione adattiva: exp(-s_i) = 1 / sigma_i^2
            precision = torch.exp(-self.log_vars[i])
            
            # Applicazione della formula di pesatura:
            # diff = (L_i / sigma_i^2) + log(sigma_i^2)
            # Nota: il termine log(sigma_i^2) funge da regolarizzatore per evitare che la rete aumenti sigma all'infinito per annullare la loss.
            diff = 0.5 * (precision * loss_dict[key] + self.log_vars[i])
            
            total_loss += diff
            
        return total_loss