import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from rotations import *


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
# Loss functions for training
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


# Rotation Loss
def geodesic_loss(pred_rot, target_rot):
    """
    Formula: theta = acos((tr(R_pred @ R_target^T) - 1) / 2)
    pred_rot, target_rot: [..., 3, 3]
    """
    # Let's calculate the "difference" matrix R_diff = R1 * R2^T
    R_diff = torch.matmul(pred_rot, target_rot.transpose(-1, -2))
    
    # Let's calculate the trace of the resulting matrix
    tr = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    
    # Let's apply the formula for the angle (in radians)
    argument = (tr - 1) / 2.0
    argument = torch.clamp(argument, -1.0 + 1e-7, 1.0 - 1e-7)
    
    loss = torch.acos(argument)

    return loss.mean()

def bone_vector_loss(pred_rot_mats, joints_3d_pred):
    """
    pred_rot_mats: [B, T, 16, 3, 3] (Output del Rotation Head)
    joints_3d_pred: [B, T, 17, 3] (Output del 3D Branch)
    """
    # 1. Otteniamo i vettori osso dal branch 3D e normalizziamoli
    # target_vectors: [B, T, 16, 3]
    target_vectors = get_h36m_bones(joints_3d_pred)
    target_vectors = F.normalize(target_vectors, p=2, dim=-1)

    # 2. Prepariamo la T-Pose per il broadcasting
    # Da [16, 3] a [1, 1, 16, 3, 1] (vettori colonna per matmul)
    v_t = get_t_pose_h36m().view(1, 1, 16, 3, 1)

    # 3. Ruotiamo i vettori della T-Pose usando le matrici predette
    # [B, T, 16, 3, 3] @ [1, 1, 16, 3, 1] -> [B, T, 16, 3, 1]
    pred_vectors = torch.matmul(pred_rot_mats, v_t)
    
    # Torniamo a [B, T, 16, 3]
    pred_vectors = pred_vectors.squeeze(-1)

    # 4. Calcoliamo la Loss (L1 è più robusta agli outlier)
    loss = F.l1_loss(pred_vectors, target_vectors)
        
    return loss


def soft_argmax_coords(heatmaps, frame_size=(352, 640), temperature=0.01):
    """
    Versione DIFFERENZIABILE del calcolo coordinate.
    
    Args:
        heatmaps: [N, 1, H, W] - Output della deconv
        frame_size: (height, width) originale
        temperature: Controlla quanto la distribuzione è "appuntita"
    """
    N, C, H, W = heatmaps.shape
    device = heatmaps.device
    H_full, W_full = frame_size

    # 1. Applichiamo Softmax spaziale con temperatura
    # Portiamo [N, 1, H, W] -> [N, H*W] per la softmax
    softmax_attention = F.softmax(heatmaps.view(N, -1) / temperature, dim=-1)
    softmax_attention = softmax_attention.view(N, H, W) # [N, H, W]

    # 2. Creiamo le griglie di coordinate (normalizzate)
    # x: [0, W-1] -> [-1, 1]
    # y: [0, H-1] -> [-H/W, H/W] per mantenere l'aspect ratio
    pos_x = torch.linspace(-1, 1, steps=W, device=device)
    pos_y = torch.linspace(-1, 1, steps=H, device=device) * (H - 1) / (W - 1)
    
    # Creiamo le matrici coordinate [H, W]
    grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')

    # 3. Calcoliamo il valore atteso (media pesata) delle coordinate
    # Sommiamo su H e W: ogni pixel contribuisce in base alla sua intensità nella heatmap
    expected_x = torch.sum(softmax_attention * grid_x, dim=(-2, -1))
    expected_y = torch.sum(softmax_attention * grid_y, dim=(-2, -1))

    return torch.stack([expected_x, expected_y], dim=-1)


# Uncertainty Heatmap Loss 
def aleatoric_heatmap_loss(mu, log_var, target_data, frame_size=(352, 640)):
    """
    Calcola la loss con incertezza aleatoria per pixel.
    
    Args:
        mu: Heatmap predetta [B, T, J, H, W]
        log_var: Log-varianza predetta [B, T, J, H, W] (s nella formula)
        target: Heatmap di riferimento (GT) [B, T, J, H, W]
    """
    H_full, W_full = frame_size
    B, T, J, H, W = mu.shape

    # 1. Split Target: Coordinate e Confidence
    target_coords = target_data[..., :2]  # [B, T, J, 2]
    target_conf = target_data[..., 2]    # [B, T, J] (Confidence ViTPose)

    # 2. Applichiamo Soft-Argmax per ottenere le coordinate predette (x, y)
    mu_flat = mu.reshape(B * T * J, 1, H, W)
    coords_pred = soft_argmax_coords(mu_flat, frame_size=frame_size)
    coords_pred = coords_pred.view(B, T, J, 2)

    scale = W_full / 2.0
    mse_per_joint = torch.sum(((coords_pred - target_coords) * scale)**2, dim=-1)

    # 1. CLAMPING: Impedisce alla varianza di collassare a zero (o log_var a -inf)
    # Un valore tra -5 e 5 di solito è un buon range di stabilità
    log_var = torch.clamp(log_var, min=-2.0, max=5.0)
    
    precision = torch.exp(-log_var)

    loss_unweighted = 0.5 * (precision * mse_per_joint + log_var)

    # 5. Applichiamo la confidence di ViTPose come peso
    # Questo moltiplica la loss del giunto per la sua "qualità" originale
    weighted_loss = loss_unweighted * target_conf
    
    return torch.mean(weighted_loss)


# ------------------------
class MultiTaskLossWrapperPretrain(nn.Module):
    def __init__(self, total_epochs, num_tasks=7):
        super().__init__()
        self.total_epochs = total_epochs
        self.logits = nn.Parameter(torch.ones(num_tasks))
        self.keys = ['mpjpe', 'bone', 'velocity', 'acceleration', 'geodesic', 'vector', 'hm']

        #scales = torch.tensor([1,1,1,1,1,1,1]) 
        scales = torch.tensor([
            0.0061,      # mpjpe (Tuo originale)
            0.00044,     # bone  (Tuo originale)
            0.0000011,   # velocity (Tuo originale)
            0.0000010,   # acceleration (Tuo originale)

            # --- QUI FACCIAMO CHIRURGIA ---
            0.1,         # geodesic (Era 2.10: lo abbassiamo per dare 20x più peso)
            0.05,        # vector   (Era 0.61: lo abbassiamo per dare 12x più peso)
            0.05         # hm       (Era 5.5: lo abbassiamo di 100 volte!)
        ])
        self.register_buffer('loss_scales', scales)

    def forward(self, loss_dict, current_epoch):
        losses = torch.stack([loss_dict[k] for k in self.keys])
        normalized_losses = losses / (self.loss_scales + 1e-8)
        
        epoch = current_epoch + 1
        mid_start = self.total_epochs // 3  # Es. Epoca 20
        ramp_duration = 10                  # Transizione di 10 epoche
        
        # --- 1. Calcolo del Gate Graduale ---
        if epoch < mid_start:
            gate = 0.0
        elif epoch < mid_start + ramp_duration:
            # Sale linearmente da 0 a 1 in 10 epoche
            gate = (epoch - mid_start) / ramp_duration
        else:
            gate = 1.0

        # --- 2. Calcolo del Floor Dinamico ---
        # Partiamo da 0.8 e scendiamo a 0.5 verso la fine per dare più peso ai dettagli
        if epoch < mid_start:
            floor = 0.8
        else:
            # Scende gradualmente da 0.8 a 0.5 dopo il mid_point
            progress = min(1.0, (epoch - mid_start) / (self.total_epochs - mid_start))
            floor = 0.8 - (0.3 * progress)

        # --- 3. Importanza con Softmax + Floor ---
        softmax_weights = F.softmax(self.logits, dim=0)
        importance = softmax_weights * (1 - floor)
        importance[0] = importance[0] + floor # MPJPE dominante
        
        # Applichiamo il gate ai task secondari
        importance[4] *= gate
        importance[5] *= gate

        # 4. Total Loss
        total_loss = torch.sum(importance * normalized_losses)
        
        # 5. Pesi effettivi per logging
        effective_weights = importance / (self.loss_scales + 1e-8)
        
        return total_loss, effective_weights
    
class MultiTaskLossWrapperSelfSupervised(nn.Module):
    def __init__(self, num_tasks=6):
        super().__init__()
        # self.log_vars rappresenta 's' nella formula: s = log(sigma^2).
        # Inizializzato a 0, il che significa sigma^2 = 1, e il peso iniziale exp(-0) = 1.
        # È un nn.Parameter, quindi l'ottimizzatore lo aggiornerà durante il backprop.
        #self.log_vars = nn.Parameter(torch.zeros(6))
        initial_log_vars = torch.tensor([
            -9.0,  # bone
            -13.0, # velocity
            -10.0, # acceleration
            -0.5,  # geodesic
            -2.5,  # vector
            0.5    # heatmap (partiamo più cauti)
        ])
        self.log_vars = nn.Parameter(initial_log_vars)

    def forward(self, loss_dict):
        """
        Riceve un dizionario dove ogni chiave è un task e ogni valore è la loss scalare (L_i).
        """
        total_loss = 0
        keys = ['bone', 'velocity', 'acceleration', 'geodesic', 'vector', 'hm']
        
        for i, key in enumerate(keys):
            current_log_var = self.log_vars[i]
            # Calcolo della precisione adattiva: exp(-s_i) = 1 / sigma_i^2
            precision = torch.exp(-current_log_var)
            
            # Applicazione della formula di pesatura:
            # diff = (L_i / sigma_i^2) + log(sigma_i^2)
            # Nota: il termine log(sigma_i^2) funge da regolarizzatore per evitare che la rete aumenti sigma all'infinito per annullare la loss.
            diff = 0.5 * (precision * loss_dict[key] + current_log_var)
            
            total_loss += diff
            
        return total_loss