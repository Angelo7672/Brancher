import os
import json
import torch
import torch.nn as nn
import shutil
from torchvision import transforms
from torchvision.io import read_video
from mmpose.apis import init_model
from config import Config

# === CONFIGURAZIONE E COSTANTI ===
W_VIDEO = 416
H_VIDEO = 640
V2_DIR = "/Experiments/angeloa00/SportsPose_reprocessed_v2"
V3_DIR = "/Experiments/angeloa00/SportsPose_reprocessed_v3"

# === 2D Pose Extractor Model ===
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
        feats = self.backbone(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        x = self.deconv_layer(feats)
        x = self.final_layer(x)
        return x

# === Keypoint Extractor Wrapper ===
class KeypointExtractor:
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

    def get_heatmaps(self, frames):
        """ frames: [T, H, W, 3] uint8 -> heatmaps: [T, J, H_out, W_out] """
        x = frames.permute(0, 3, 1, 2).float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(self.model_h, self.model_w), mode='bilinear', align_corners=False)
        x = self.normalizer(x).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(x)
        return heatmaps

# === Funzione di Estrazione Coordinate ===
def extract_coordinates(heatmaps, video_w, video_h):
    """
    Converte le heatmaps in coordinate (x, y, conf) riscalate alla risoluzione video.
    heatmaps: [T, J, Hh, Wh]
    """
    T, J, Hh, Wh = heatmaps.shape
    all_kpts = []

    # Portiamo tutto su CPU per l'elaborazione delle coordinate
    heatmaps = torch.clamp(heatmaps, min=0.0, max=1.0).cpu()

    for t in range(T):
        frame_kpts = torch.zeros((J, 3))
        for j in range(J):
            hmap = heatmaps[t, j]
            conf, idx = torch.max(hmap.view(-1), 0)
            y_h, x_h = divmod(int(idx.item()), Wh)

            # Riscalatura proporzionale
            x_v = (x_h / Wh) * video_w
            y_v = (y_h / Hh) * video_h
            frame_kpts[j] = torch.tensor([x_v, y_v, conf])
            
        all_kpts.append(frame_kpts.tolist())
    
    return all_kpts

# === MAIN PIPELINE ===
def main():
    extractor = KeypointExtractor()
    os.makedirs(V3_DIR, exist_ok=True)

    for sample_folder in sorted(os.listdir(V2_DIR)):
        folder_v2 = os.path.join(V2_DIR, sample_folder)
        if not os.path.isdir(folder_v2):
            continue

        video_path = os.path.join(folder_v2, "video.mp4")
        pose3d_path = os.path.join(folder_v2, "pose3d.json")

        if not os.path.isfile(video_path):
            continue

        print(f"\n🚀 Processing: {sample_folder}")
        folder_v3 = os.path.join(V3_DIR, sample_folder)
        os.makedirs(folder_v3, exist_ok=True)

        # 1. Caricamento Video
        video_tensor, _, _ = read_video(video_path, output_format="THWC", pts_unit="sec")
        
        # 2. Inferenza Heatmaps
        heatmaps = extractor.get_heatmaps(video_tensor) # [T, J, Hh, Wh]
        
        # 3. Estrazione Keypoints 2D (Coordinate)
        keypoints_2d = extract_coordinates(heatmaps, W_VIDEO, H_VIDEO)

        # 4. Salvataggio Heatmaps (FP16 per risparmiare spazio)
        torch.save(heatmaps.half().cpu(), os.path.join(folder_v3, "heatmaps.pt"))
        
        # 5. Salvataggio JSON Pose 2D
        with open(os.path.join(folder_v3, "pose2d.json"), "w") as f:
            json.dump(keypoints_2d, f)

        # 6. Copia Pose 3D originale
        if os.path.isfile(pose3d_path):
            shutil.copy(pose3d_path, os.path.join(folder_v3, "pose3d.json"))

        print(f"✅ Done: {sample_folder} (Frames: {len(keypoints_2d)})")

if __name__ == "__main__":
    main()
