from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torchinfo import summary
from mmpose.apis import init_model
import os

# === YOLOv8 model ===
person_detector = YOLO("yolov8n.pt")

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

# === Trasformazione input HRNet ===
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
def extract_keypoints_with_confidence(heatmaps, input_size=(256, 192), orig_size=(0, 0), conf_thresh=0.2):
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

# === Disegna scheletro COCO su immagine BGR ===
def draw_skeleton(frame, keypoints, skeleton, color=(0, 255, 0)):
    for pt in keypoints:
        if pt is not None:
            x, y = pt
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for i, j in skeleton:
        if keypoints[i] is not None and keypoints[j] is not None:
            xi, yi = keypoints[i]
            xj, yj = keypoints[j]
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2)
    return frame

coco_skeleton = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6)
]

# === MAIN ===
if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_file = 'AR17/code/keypoints_module/hrnet/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    model = HR(config_file, checkpoint_file, device=device).to(device)
    model.eval()

    summary(model, input_size=(1, 3, 256, 192), device=device)

    input_path = 'AR17/code/images/IMG_3962.JPG'
    image = cv2.imread(input_path)

    # === Step 1: Detect people ===
    results = person_detector(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()

    # === Step 2: For each person box, extract keypoints ===
    for i, box in enumerate(boxes):
        if int(cls[i]) != 0:  # COCO class 0 == person
            continue
        x1, y1, x2, y2 = map(int, box)
        person_crop = image[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue
        input_tensor = preprocess(person_crop).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            heatmaps = output[0].detach().cpu().numpy()
            keypoints_local = extract_keypoints_with_confidence(heatmaps, orig_size=person_crop.shape[:2])
            # keypoints in full image coords
            keypoints_global = [(pt[0] + x1, pt[1] + y1) if pt is not None else None for pt in keypoints_local]
            # draw on original image (visual check)
            image = draw_skeleton(image, keypoints_global, coco_skeleton)

    # save an annotated visualization of detections
    cv2.imwrite("AR17/code/images/output_pose_hr_yolo.jpg", image)