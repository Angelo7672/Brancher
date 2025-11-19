import torch

class KeypointAugmentor:
    """
    Applica augmentation sui keypoints 2D (x, y, conf).
    Ogni frame ha 15% di probabilità di essere perturbato.
    Dentro quel frame, il 45% dei joint subisce rumore gaussiano.
    La confidence viene aggiornata in base alla distanza dal punto originale.
    """
    def __init__(self, frame_prob=0.15, joint_prob=0.45, noise_std=0.002):
        self.frame_prob = frame_prob
        self.joint_prob = joint_prob
        self.noise_std = noise_std

    def dis2conf(self, dis, a=1.0, b=-0.3, m=0.0, s=0.05):
        """Riduce la confidenza in base alla distanza dal punto originale."""
        f = a / (dis + a) + b * dis
        shift = torch.randn_like(dis) * s + m
        return (f + shift).clamp(0, 1)

    @torch.no_grad()
    def __call__(self, kpts):
        """
        kpts: torch.Tensor (B,T,J,3)
        Ritorna: torch.Tensor (B,T,J,3) augmentato
        """
        B, T, J, _ = kpts.shape
        device = kpts.device

        # Maschera frame (ogni frame per ogni batch ha prob 15%)
        frame_mask = (torch.rand(B, T, 1, 1, device=device) < self.frame_prob)
        # Maschera joint (45% di joint per frame)
        joint_mask = (torch.rand(B, T, J, 1, device=device) < self.joint_prob)
        # Maschera finale
        full_mask = frame_mask & joint_mask

        # Rumore gaussiano su x,y
        noise = torch.randn(B, T, J, 2, device=device) * self.noise_std
        delta = noise * full_mask  # Applica rumore solo dove serve

        # Aggiorna coordinate
        xy_noisy = kpts[..., :2] + delta

        # Calcola nuova confidenza
        dis = torch.norm(delta, dim=-1)  # distanza del rumore (B,T,J)
        conf_new = self.dis2conf(dis).unsqueeze(-1)

        # Ricombina x,y,conf
        kpts_noisy = torch.cat((xy_noisy, conf_new), dim=-1)

        # Mantieni la confidenza originale dove non è stato applicato rumore
        kpts_noisy[..., 2] = torch.where(
            full_mask.squeeze(-1),
            conf_new.squeeze(-1),
            kpts[..., 2]
        )

        return kpts_noisy