import torch
import torch.nn as nn
from mmpose.apis import init_model
from PIL import Image
from torchvision import transforms
from torchinfo import summary
import cv2
import numpy as np

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))  # se hai una GPU


class HR(nn.Module):
    def __init__(self, config_file, checkpoint_file, device='cpu', freeze_backbone=False):
        super(HR, self).__init__()
        
        # Carica il modello HR pre-addestrato
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

def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)  # [1, 3, 256, 192]

def extract_keypoints_with_confidence(heatmaps, input_size=(256, 192), orig_size=(0, 0), conf_thresh=0.2):
    input_h, input_w = input_size
    heatmap_h, heatmap_w = heatmaps.shape[1:]
    scale_x = (input_w / heatmap_w) * (orig_size[1] / input_w)
    scale_y = (input_h / heatmap_h) * (orig_size[0] / input_h)

    keypoints = []
    for heatmap in heatmaps:
        conf = np.max(heatmap)
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x_orig = int(x * scale_x)
        y_orig = int(y * scale_y)
        if conf >= conf_thresh:
            keypoints.append((x_orig, y_orig))
        else:
            keypoints.append(None)  # Punto non affidabile
    return keypoints


def draw_keypoints(frame, keypoints):
    for (x, y) in keypoints:
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    return frame

coco_skeleton = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12),
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (1, 2),
    (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6)
]

def draw_skeleton(frame, keypoints, skeleton, color=(0, 255, 0)):
    for pt in keypoints:
        if pt is not None:
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)
    for i, j in skeleton:
        if keypoints[i] is not None and keypoints[j] is not None:
            cv2.line(frame, keypoints[i], keypoints[j], color, 2)
    return frame

if __name__ == '__main__':

    # Sostituisci con i percorsi locali o URL corretti
    config_file = 'keypoints_module/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-wholebody-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_256x192-643e18cb_20200922.pth'
    #checkpoint_file = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_posetrack18_256x192-b5d9b3f1_20211130.pth"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HR(config_file, checkpoint_file, device=device, freeze_backbone=True).to(device)
    model.eval()

    summary(model, input_size=(1, 3, 256, 192), device=device)

    # === VIDEO ===
    input_video_path = 'video/video.mp4'
    output_video_path = 'video/output_pose_video_hr.mp4'
    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = preprocess(frame).to(device)
            output = model(input_tensor)
            heatmaps = output[0].detach().cpu().numpy()
            keypoints = extract_keypoints_with_confidence(heatmaps, orig_size=frame.shape[:2])
            frame_with_kps = draw_skeleton(frame, keypoints, coco_skeleton)

            out_video.write(frame_with_kps)

    cap.release()
    out_video.release()
    print(f"[✓] Video salvato in: {output_video_path}")