from ultralytics import YOLO
import torch
import torch.nn as nn
from mmpose.apis import init_model
from PIL import Image
from torchvision import transforms
from torchinfo import summary
import cv2
import numpy as np
import os

# === YOLOv8 model ===
person_detector = YOLO("yolov8n.pt")


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

def preprocess(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)


def extract_keypoints(heatmaps, input_size=(256, 192), orig_size=(0, 0), conf_thresh=0.4):
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
            keypoints.append(None)
    return keypoints


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
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
    for i, j in skeleton:
        if keypoints[i] is not None and keypoints[j] is not None:
            cv2.line(frame,
                     (int(keypoints[i][0]), int(keypoints[i][1])),
                     (int(keypoints[j][0]), int(keypoints[j][1])),
                     color, 2)
    return frame


if __name__ == '__main__':
    config_file = 'AR17/code/keypoints_module/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = HR(config_file, checkpoint_file, device=device, freeze_backbone=True).to(device)
    model.eval()

    summary(model, input_size=(1, 3, 256, 192), device=device)

    # === INPUT VIDEO ===
    input_video = "AR17/code/video/IMG_3990.mp4"
    output_video = "AR17/code/video/output_pose_hr_yolo.mp4"

    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, fps, (W, H))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # === Step 1: Detect people ===
        results = person_detector(frame, verbose=False)
        if len(results) == 0 or results[0].boxes is None:
            out.write(frame)
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls = results[0].boxes.cls.cpu().numpy()

        # === Step 2: For each person box, extract keypoints ===
        for i, box in enumerate(boxes):
            if int(cls[i]) != 0:    # Only process person class
                continue
            x1, y1, x2, y2 = map(int, box)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            input_tensor = preprocess(person_crop).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                heatmaps = output[0].detach().cpu().numpy()
                keypoints_local = extract_keypoints(heatmaps, orig_size=person_crop.shape[:2])
                valid_count = sum(1 for pt in keypoints_local if pt is not None)
                if valid_count < 9:  # Require at least 8 valid keypoints
                    continue
                # keypoints in full image coords
                keypoints_global = [
                    (pt[0] + x1, pt[1] + y1) if pt is not None else None
                    for pt in keypoints_local
                ]
                frame = draw_skeleton(frame, keypoints_global, coco_skeleton)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Output salvato in: {output_video}")