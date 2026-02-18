import torch
import torch.nn.functional as F
from config import Config

cfg = Config()

# H36M Notation: 0-Pelvis, 1-R-Hip, 2-R-Knee, 3-R-Ankle, 4-L-Hip, 5-L-Knee, 6-L-Ankle, 7-Spine, 8-Thorax, 9-Neck,
# 10-Head, 11-L-Shoulder, 12-L-Elbow, 13-L-Wrist, 14-R-Shoulder, 15-R-Elbow, 16-R-Wrist

def get_h36m_bones(joints_3d):
    """
    Extracts bone vectors following the H36M hierarchy.
    Input joints_3d: [B, T, 17, 3]
    Output bones: [B, T, 16, 3]
    """
    # Definition of pairs (parent, child) for H36M (17 joints)
    pairs = [
        (0, 1), (1, 2), (2, 3),    # Right Leg
        (0, 4), (4, 5), (5, 6),    # Left Leg
        (0, 7), (7, 8),            # Spine
        (8, 9), (9, 10),           # Neck and Head
        (8, 11), (11, 12), (12, 13), # Left Arm
        (8, 14), (14, 15), (15, 16)  # Right Arm
    ]
    
    bones = []
    for parent, child in pairs:
        # Bone Vector = Child Joint - Parent Joint
        bones.append(joints_3d[:, :, child] - joints_3d[:, :, parent])
        
    return torch.stack(bones, dim=2) # [B, T, 16, 3]

def get_t_pose_h36m():
    """
    Create the reference unit vector star for H36M.
    Orientation System: Y=up, Z=right, X=forward.
    The subject's head looks in the direction +X.
    """
    # 16 bones, 3 coordinates
    t_info = torch.zeros(16, 3, device=cfg.device)
    
    # --- Axial Directions ---
    UP = torch.tensor([0., 1., 0.], device=cfg.device)
    DOWN = torch.tensor([0., -1., 0.], device=cfg.device)
    LEFT = torch.tensor([0., 0., -1.], device=cfg.device)
    RIGHT = torch.tensor([0., 0., 1.], device=cfg.device)
    
    # 0,1,2: Right Leg
    t_info[0] = RIGHT # Pelvis -> R_Hip
    t_info[1] = DOWN  # R_Thigh
    t_info[2] = DOWN  # R_Calf
    
    # 3,4,5: Left Leg
    t_info[3] = LEFT  # Pelvis -> L_Hip
    t_info[4] = DOWN  # L_Thigh
    t_info[5] = DOWN  # L_Calf
    
    # 6,7,8,9: Spine, Neck, Head
    t_info[6] = UP    # Pelvis -> Spine
    t_info[7] = UP    # Spine -> Thorax
    t_info[8] = UP    # Thorax -> Neck
    t_info[9] = UP    # Neck -> Head
    
    # 10,11,12: Left Arm
    t_info[10] = LEFT # Thorax -> L_Shoulder
    t_info[11] = LEFT # L_Upper_Arm
    t_info[12] = LEFT # L_Lower_Arm
    
    # 13,14,15: Right Arm
    t_info[13] = RIGHT # Thorax -> R_Shoulder
    t_info[14] = RIGHT # R_Upper_Arm
    t_info[15] = RIGHT # R_Lower_Arm
    
    return F.normalize(t_info, p=2, dim=-1)


def get_rotation_between_vectors(a, b):
    """
    Calculate the rotation matrix that aligns vector 'a' to vector 'b'.
    a: [16, 3] (T-Pose bones)
    b: [B, T, 16, 3] (Predicted 3D bones)
    """
    # Normalization
    a = F.normalize(a, p=2, dim=-1) # a / ||a||
    b = F.normalize(b, p=2, dim=-1) # b / ||b||
    
    # Let's expand to broadcasting: [1, 1, 16, 3]
    a = a.unsqueeze(0).unsqueeze(0)
    
    v = torch.cross(a, b, dim=-1)   # v = a x b = ||a|| ||b|| sen(theta)
    c = torch.sum(a * b, dim=-1) # c = a * b = ||a|| ||b|| cos(theta)
    s = torch.norm(v, p=2, dim=-1) # s = ||v||
    
    # Handling the case where vectors are already aligned (avoid division by zero)
    eps = 1e-8
    factor = (1 - c) / (s**2 + eps)
    
    # Skew-symmetric matrix components
    vx = v[..., 0]
    vy = v[..., 1]
    vz = v[..., 2]
    
    # Let's build the matrix K (skew-symmetric)
    # [0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]
    zeros = torch.zeros_like(vx)
    K = torch.stack([
        torch.stack([zeros, -vz, vy], dim=-1),
        torch.stack([vz, zeros, -vx], dim=-1),
        torch.stack([-vy, vx, zeros], dim=-1)
    ], dim=-2) # [B, T, 16, 3, 3]
    
    # Rodrigues formula: R = I + K + K^2 * (1 - c) / (s**2)
    I = torch.eye(3, device=a.device).reshape(1, 1, 1, 3, 3)
    R = I + K + torch.matmul(K, K) * factor.unsqueeze(-1).unsqueeze(-1)
    
    return R