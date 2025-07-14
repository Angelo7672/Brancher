import torch
import torch.nn as nn
from mmpose.apis import init_model
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # se hai una GPU


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

if __name__ == '__main__':

    # Sostituisci con i percorsi locali o URL corretti
    config_file = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViTPoseBase(config_file, checkpoint_file, device=device, freeze_backbone=True).to(device)
    model.eval()


    img = Image.open('image.png').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        # Add normalization if required by your model
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)  # Shape: [1, 3, 256, 192]

    output = model(input_tensor)

    print('Output shape:', output.shape)  # Tipicamente [1, 17, 64, 48]

    # Post-process: estrai keypoints direttamente dai heatmap
    heatmaps = output[0].detach().cpu().numpy()  # [num_joints, H, W]
    
    # Estrai la posizione (x, y) del massimo per ciascun joint
    keypoints = []
    for heatmap in heatmaps:
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        keypoints.append((x, y))  # attenzione: (x, y) in coordinate heatmap (es. 48x64)
    
    # Dimensioni di input del modello
    input_h, input_w = 256, 192
    heatmap_h, heatmap_w = heatmaps.shape[1:]
    
    # Fattore di upscaling dal heatmap all'immagine modello
    scale_heatmap_to_input_x = input_w / heatmap_w  # es. 192 / 48 = 4.0
    scale_heatmap_to_input_y = input_h / heatmap_h  # es. 256 / 64 = 4.0
    
    # Carica immagine originale
    orig_img = cv2.imread('image.png')
    orig_h, orig_w = orig_img.shape[:2]
    
    # Scala da immagine modello a immagine originale
    scale_input_to_orig_x = orig_w / input_w
    scale_input_to_orig_y = orig_h / input_h
    
    # Combina scaling totale dal heatmap all’immagine originale
    scale_x = scale_heatmap_to_input_x * scale_input_to_orig_x
    scale_y = scale_heatmap_to_input_y * scale_input_to_orig_y
    
    # Ridimensiona e disegna i keypoint
    for (x, y) in keypoints:
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)
        cv2.circle(orig_img, (x_orig, y_orig), 4, (0, 255, 0), -1)  # cerchio verde
    
    # Salva immagine con keypoints disegnati
    cv2.imwrite('image_with_keypoints.png', orig_img)
    print('Salvata image_with_keypoints.png')
