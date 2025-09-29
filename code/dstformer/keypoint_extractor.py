import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from mmpose.apis import init_model

# === HRNet wrapper ===
class HR(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=False):
        super(HR, self).__init__()
        self.model = init_model(config_file, checkpoint_file, device=device)
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        x = self.deconv_layer(features)
        x = self.final_layer(x)
        return x

# === ViTPose wrapper ===
class ViTPoseBase(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=False):
        super(ViTPoseBase, self).__init__()
        
        # Carica il modello ViTPose pre-addestrato
        self.model = init_model(config_file, checkpoint_file, device=device)
        
        # Se vuoi congelare il backbone
        if freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

        # Salva le componenti principali
        self.backbone = self.model.backbone
        self.deconv_layer = self.model.head.deconv_layers
        self.final_layer = self.model.head.final_layer

    def forward(self, x):
        """
        x: Tensor di forma [B, 3, H, W]
        Output: heatmap [B, num_joints, H_out, W_out]
        """
        # Estrai feature dal backbone
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]
        
        # Passa attraverso deconv + final layer (testa)
        x = self.deconv_layer(features)
        x = self.final_layer(x)
        return x

# === Trasformazione input ===
def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# === Estrazione keypoint da heatmap ===
def extract_keypoints(heatmaps, input_size=(256, 192), orig_size=(0, 0), conf_thresh=0.5):
    input_h, input_w = input_size
    heatmap_h, heatmap_w = heatmaps.shape[1:]
    scale_x = (orig_size[1] / heatmap_w)
    scale_y = (orig_size[0] / heatmap_h)

    keypoints = []
    for heatmap in heatmaps:
        conf = np.max(heatmap)
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x_orig = x * scale_x
        y_orig = y * scale_y
        if conf >= conf_thresh:
            keypoints.append((float(x_orig), float(y_orig)))
        else:
            keypoints.append(None)
    return keypoints