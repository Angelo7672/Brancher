import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.io import read_video
from mmpose.apis import init_model
from config import Config
import os
import shutil


# === 2D Pose Extractor (senza YOLO) ===
class PE(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=True):
        super().__init__()

        self.model = init_model(config_file, checkpoint_file, device=device)

        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        """
        x: [T, 3, H_model, W_model]
        out: [T, J, H_out, W_out]
        """
        feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        x = self.deconv_layer(feats)
        x = self.final_layer(x)
        return x


# === Keypoint Extractor SEMPLIFICATO ===
class KeypointExtractorSimple:
    def __init__(self):
        cfg = Config()

        self.device = torch.device(cfg.device)
        self.model_h, self.model_w = cfg.pe_size
        self.model = PE(cfg.config_file, cfg.checkpoint_file, device=self.device).to(self.device)
        self.model.eval()

        self.normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, frames):
        """
        frames: [T, H, W, 3] uint8
        """
        T = len(frames)

        # Convert to float and permute
        x = frames.permute(0, 3, 1, 2).float() / 255.0  # [T,3,H,W]

        # Resize to model input size
        x = torch.nn.functional.interpolate(
            x,
            size=(self.model_h, self.model_w),
            mode='bilinear',
            align_corners=False
        )

        # Normalize
        x = self.normalizer(x)

        x = x.to(self.device)

        with torch.no_grad():
            heatmaps = self.model(x)

        return heatmaps  # [T, J, H_out, W_out]


# === USO ===
num_joints = 17
device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = KeypointExtractorSimple()

v2_dir = "/Experiments/angeloa00/AirPose/train"
v3_dir = "/Experiments/angeloa00/AirPose_v2/train"

# === Itera tutte le sottocartelle in v2 ===
for sample_folder in sorted(os.listdir(v2_dir)):
    folder_v2 = os.path.join(v2_dir, sample_folder)
    if not os.path.isdir(folder_v2):
        continue
    
    print(f"Processing folder: {sample_folder}")
    
    video_path = os.path.join(folder_v2, "video.mp4")
    pose3d_path = os.path.join(folder_v2, "pose3d.json")
    
    # Controlla che il video esista
    if not os.path.isfile(video_path):
        print(f"  Video missing in {folder_v2}, skipping.")
        continue
    
    # Crea cartella corrispondente in v3
    folder_v3 = os.path.join(v3_dir, sample_folder)
    os.makedirs(folder_v3, exist_ok=True)
    
    # Carica video
    print("  Loading video...")
    video_tensor, _, _ = read_video(video_path, output_format="THWC", pts_unit="sec")
    print("  Video shape:", video_tensor.shape)
    
    # Inferenzia heatmaps
    print("  Running keypoint extractor...")
    heatmaps = extractor(video_tensor)
    print("  Heatmap shape:", heatmaps.shape)
    
    # Salva heatmaps in FP16
    heatmap_path = os.path.join(folder_v3, "heatmaps.pt")
    torch.save(heatmaps.half(), heatmap_path)
    print(f"  Saved heatmaps to {heatmap_path}")
    
    # Copia pose3d.json
    if os.path.isfile(pose3d_path):
        shutil.copy(pose3d_path, os.path.join(folder_v3, "pose3d.json"))
        print("  Copied pose3d.json")
    
print("All videos processed.")
