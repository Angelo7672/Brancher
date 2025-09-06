from ultralytics import YOLO
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms
from torchinfo import summary
from mmpose.apis import init_model
import os
import pathlib
import uuid

# === YOLOv8 model ===
person_detector = YOLO("yolov8n.pt")  # o yolov8s.pt

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
            keypoints.append((float(x_orig), float(y_orig), float(conf)))
        else:
            keypoints.append(None)
    return keypoints

# === Disegna scheletro COCO su immagine BGR ===
def draw_skeleton(frame, keypoints, skeleton, color=(0, 255, 0)):
    for pt in keypoints:
        if pt is not None:
            x, y, _ = pt
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for i, j in skeleton:
        if keypoints[i] is not None and keypoints[j] is not None:
            xi, yi, _ = keypoints[i]
            xj, yj, _ = keypoints[j]
            cv2.line(frame, (int(xi), int(yi)), (int(xj), int(yj)), color, 2)
    return frame

coco_skeleton = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6)
]

# === Render stickman su sfondo neutro (RGB PIL image) ===
def render_stickman(keypoints, skeleton, canvas_size=(256,192), line_width=4, joint_radius=3):
    W, H = canvas_size[0], canvas_size[1]  # we'll assume input keypoints are in this coordinate frame
    img = Image.new("RGB", (W, H), (255,255,255))
    draw = ImageDraw.Draw(img)
    # line segments (skeleton)
    for (i, j) in skeleton:
        if keypoints[i] is not None and keypoints[j] is not None:
            xi, yi, _ = keypoints[i]
            xj, yj, _ = keypoints[j]
            draw.line([(xi, yi), (xj, yj)], fill=(0,0,0), width=line_width)
    # joints
    for pt in keypoints:
        if pt is not None:
            x, y, conf = pt
            r = joint_radius
            draw.ellipse([x-r, y-r, x+r, y+r], fill=(0,0,0))
    return img

# === Helper per normalizzare i keypoints sulla crop (dimensione target) ===
def normalize_keypoints_to_size(keypoints_local, src_size, target_size=(256,192)):
    src_h, src_w = src_size
    tgt_w, tgt_h = target_size
    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h
    normalized = []
    for pt in keypoints_local:
        if pt is None:
            normalized.append(None)
        else:
            x, y, conf = pt
            nx = x * scale_x
            ny = y * scale_y
            normalized.append((nx, ny, conf))
    return normalized

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

    input_path = 'AR17/code/images/image_mon.png'
    image = cv2.imread(input_path)

    # output dir for stickman images to feed to SV3D
    out_stick_dir = pathlib.Path("AR17/code/stickman_inputs")
    out_stick_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Detect people ===
    results = person_detector(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    cls = results[0].boxes.cls.cpu().numpy()

    # === Step 2: For each person box, extract keypoints and render stickman ===
    saved_inputs = []  # list of dicts with metadata
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
            keypoints_global = [(pt[0] + x1, pt[1] + y1, pt[2]) if pt is not None else None for pt in keypoints_local]
            # draw on original image (visual check)
            image = draw_skeleton(image, keypoints_global, coco_skeleton)

            # normalize local keypoints to target canvas size for stickman rendering
            norm_kp = normalize_keypoints_to_size(keypoints_local, src_size=person_crop.shape[:2], target_size=(256,192))
            stick_img = render_stickman(norm_kp, coco_skeleton, canvas_size=(256,192))
            # save stickman image
            uid = uuid.uuid4().hex[:8]
            stick_path = out_stick_dir / f"stick_{uid}.png"
            stick_img.save(str(stick_path))
            saved_inputs.append({
                "person_idx": i,
                "box": (x1,y1,x2,y2),
                "stick_path": str(stick_path),
                "crop_shape": person_crop.shape[:2],
                "keypoints_local": keypoints_local
            })

    # save an annotated visualization of detections
    cv2.imwrite("AR17/code/images/output_pose_hr_yolo.jpg", image)
    print(f"Saved {len(saved_inputs)} stickman inputs to {out_stick_dir}")

    # --- Ora puoi prendere le immagini in out_stick_dir e passarle a SV3D ---
    print("Next steps (examples):")
    print("1) If you have the SV3D repo locally, run its inference script on each stickman PNG.")
    print("   Example (pseudocommand, adapt to your SV3D repo):")
    print("   $ python path/to/sv3d/scripts/infer.py --input AR17/code/stickman_inputs/stick_XXXX.png --output out_dir --num-views 2 --camera-trajectory orbit")
    print()
    print("2) Or use Stability/SV3D service (if available). Example pseudocode using their SDK/HTTP API:")
    print("   # set STABILITY_API_KEY env var, then call the endpoint with the stickman PNG as conditioning image")
    print("   # refer to Stability / SV3D docs for exact payload and parameters (camera path, seed, scale, etc.)")
