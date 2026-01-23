import torch
import torch.nn.functional as F

# Mappatura indici COCO: 
# 0:Nose, 1:L_Eye, 2:R_Eye, 3:L_Ear, 4:R_Ear, 5:L_Sho, 6:R_Sho, 7:L_Elb, 8:R_Elb, 9:L_Wri, 10:R_Wri, 11:L_Hip, 12:R_Hip, 13:L_Kne, 14:R_Kne, 15:L_Ank, 16:R_Ank

# Mappatura ossa COCO:
# 0:Spine (mid_hip -> mid_sho)
# 1:L_Shoulder_Base (mid_sho -> 5)
# 2:L_Upper_Arm (5 -> 7)
# 3:L_Lower_Arm (7 -> 9)
# 4:R_Shoulder_Base (mid_sho -> 6)
# 5:R_Upper_Arm (6 -> 8)
# 6:R_Lower_Arm (8 -> 10)
# 7:L_Hip_Base (mid_hip -> 11)
# 8:L_Upper_Leg (11 -> 13)
# 9:L_Lower_Leg (13 -> 15)
# 10:R_Hip_Base (mid_hip -> 12)
# 11:R_Upper_Leg (12 -> 14)
# 12:R_Lower_Leg (14 -> 16)

def get_coco_bones(joints_3d):
    """
    Input: [B, T, J, 3]
    Output: [B, T, 13, 3] (Vettori delle ossa)
    """
    B, T, _, _ = joints_3d.shape
    
    # Creiamo giunti virtuali per la root e il collo
    mid_hip = (joints_3d[:, :, 11] + joints_3d[:, :, 12]) / 2      # Centro bacino
    mid_sho = (joints_3d[:, :, 5] + joints_3d[:, :, 6]) / 2        # Centro spalle
    
    bones = []
    # 1. Tronco e Testa
    bones.append(mid_sho - mid_hip)          # Spine
    
    # 2. Braccio Sinistro
    bones.append(joints_3d[:, :, 5] - mid_sho) # L_Shoulder_Base
    bones.append(joints_3d[:, :, 7] - joints_3d[:, :, 5]) # L_Upper_Arm
    bones.append(joints_3d[:, :, 9] - joints_3d[:, :, 7]) # L_Lower_Arm
    
    # 3. Braccio Destro
    bones.append(joints_3d[:, :, 6] - mid_sho) # R_Shoulder_Base
    bones.append(joints_3d[:, :, 8] - joints_3d[:, :, 6]) # R_Upper_Arm
    bones.append(joints_3d[:, :, 10] - joints_3d[:, :, 8]) # R_Lower_Arm
    
    # 4. Gamba Sinistra
    bones.append(joints_3d[:, :, 11] - mid_hip) # L_Hip_Base
    bones.append(joints_3d[:, :, 13] - joints_3d[:, :, 11]) # L_Upper_Leg
    bones.append(joints_3d[:, :, 15] - joints_3d[:, :, 13]) # L_Lower_Leg
    
    # 5. Gamba Destra
    bones.append(joints_3d[:, :, 12] - mid_hip) # R_Hip_Base
    bones.append(joints_3d[:, :, 14] - joints_3d[:, :, 12]) # R_Upper_Leg
    bones.append(joints_3d[:, :, 16] - joints_3d[:, :, 14]) # R_Lower_Leg
    
    return torch.stack(bones, dim=2) # [B, T, 13, 3]


def get_rotation_between_vectors(a, b):
    """
    Calcola la matrice di rotazione che allinea il vettore a al vettore b.
    a: [13, 3] (T-Pose bones)
    b: [B, T, 13, 3] (Predicted 3D bones)
    """
    # Normalizzazione
    a = F.normalize(a, p=2, dim=-1) # [13, 3]
    b = F.normalize(b, p=2, dim=-1) # [B, T, 13, 3]
    
    # Espandiamo a per il broadcasting: [1, 1, 13, 3]
    a = a.unsqueeze(0).unsqueeze(0)
    
    v = torch.cross(a, b, dim=-1)
    c = torch.sum(a * b, dim=-1) # Dot product
    s = torch.norm(v, p=2, dim=-1)
    
    # Gestione del caso in cui i vettori siano già allineati (evita divisione per zero)
    eps = 1e-8
    k = 1.0 / (1.0 + c + eps)
    
    # Skew-symmetric matrix components
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    
    # Costruiamo la matrice K (skew-symmetric)
    # [0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]
    zeros = torch.zeros_like(vx)
    K = torch.stack([
        torch.stack([zeros, -vz, vy], dim=-1),
        torch.stack([vz, zeros, -vx], dim=-1),
        torch.stack([-vy, vx, zeros], dim=-1)
    ], dim=-2) # [B, T, 13, 3, 3]
    
    # Formula di Rodrigues: R = I + K + K^2 * k
    I = torch.eye(3, device=a.device).reshape(1, 1, 1, 3, 3)
    R = I + K + torch.matmul(K, K) * k.unsqueeze(-1).unsqueeze(-1)
    
    return R


def get_t_pose_coco(device):
    """
    Definisce i vettori unitari di riferimento per una T-Pose.
    Sistema: Z=up, X=right, Y=forward.
    Output: [13, 3] (Vettori unitari per le 13 ossa)
    """
    # Inizializziamo a zero
    t_info = torch.zeros(13, 3).to(device)
    
    # 1. Tronco e Testa (Verticali verso l'alto +Z)
    t_info[0] = torch.tensor([0., 0., 1.])   # Spine (mid_hip -> mid_sho)
    
    # 2. Braccio Sinistro (L_Sho è a sinistra rispetto al centro, quindi X decrescente)
    t_info[1] = torch.tensor([-1., 0., 0.])  # L_Shoulder_Base (mid_sho -> L_Sho)
    t_info[2] = torch.tensor([-1., 0., 0.])  # L_Upper_Arm (L_Sho -> L_Elb)
    t_info[3] = torch.tensor([-1., 0., 0.])  # L_Lower_Arm (L_Elb -> L_Wri)
    
    # 3. Braccio Destro (R_Sho è a destra, X crescente)
    t_info[4] = torch.tensor([1., 0., 0.])   # R_Shoulder_Base (mid_sho -> R_Sho)
    t_info[5] = torch.tensor([1., 0., 0.])   # R_Upper_Arm (R_Sho -> R_Elb)
    t_info[6] = torch.tensor([1., 0., 0.])   # R_Lower_Arm (R_Elb -> R_Wri)
    
    # 4. Gamba Sinistra (Bacino -> Piedi, Verticali verso il basso -Z)
    t_info[7] = torch.tensor([-1., 0., 0.])  # L_Hip_Base (mid_hip -> L_Hip)
    t_info[8] = torch.tensor([0., 0., -1.])  # L_Upper_Leg (L_Hip -> L_Kne)
    t_info[9] = torch.tensor([0., 0., -1.]) # L_Lower_Leg (L_Kne -> L_Ank)
    
    # 5. Gamba Destra
    t_info[10] = torch.tensor([1., 0., 0.])  # R_Hip_Base (mid_hip -> R_Hip)
    t_info[11] = torch.tensor([0., 0., -1.]) # R_Upper_Leg (R_Hip -> R_Kne)
    t_info[12] = torch.tensor([0., 0., -1.]) # R_Lower_Leg (R_Kne -> R_Ank)

    # Nota: Applichiamo la normalizzazione per sicurezza su tutti i vettori
    return F.normalize(t_info, p=2, dim=-1)