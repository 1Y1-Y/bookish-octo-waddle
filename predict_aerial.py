import cv2
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import sys

CLASSES = [
    "Black Spot",
    "Black shank",
    "Healthy",
    "Leaf Spot",
    "Tobacco leaf curl disease",
    "Tobacco mosaic virus",
    "Yellow Leaf Curl Virus",
    "brown sport",
]
COLORS = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (128, 0, 128),
    (0, 165, 255),
]

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cls_model.pt")
PATCH = 224
STRIDE = 112
CONF = 0.5


def load_model():
    model = torchvision.models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model = model.cuda().eval()
    return model


TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def scan_image(model, img):
    """滑窗切片并逐块分类，返回 ({类别索引: 数量}, 总检出数)。img 为 BGR(cv2)"""
    h, w = img.shape[:2]
    counts = {}
    total = 0
    for y in range(0, h - PATCH + 1, STRIDE):
        for x in range(0, w - PATCH + 1, STRIDE):
            patch = img[y:y + PATCH, x:x + PATCH]
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            t = TF(rgb).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(t)
                prob = torch.softmax(out, dim=1)
                ci = int(out.argmax(1).item())
                conf = float(prob[0, ci].item())
            if conf < CONF:
                continue
            total += 1
            counts[ci] = counts.get(ci, 0) + 1
    return counts, total


def predict_aerial(img_path, out_path="result.jpg"):
    model = load_model()
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片:", img_path)
        return
    h, w = img.shape[:2]
    overlay = img.copy()
    stats = {c: 0 for c in CLASSES}

    for y in range(0, h - PATCH + 1, STRIDE):
        for x in range(0, w - PATCH + 1, STRIDE):
            patch = img[y:y + PATCH, x:x + PATCH]
            t = TF(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
            with torch.no_grad():
                out = model(t)
                prob = torch.softmax(out, dim=1)
                cls = int(out.argmax(1).item())
                conf = float(prob[0, cls].item())
            if conf < CONF:
                continue
            stats[CLASSES[cls]] += 1
            cv2.rectangle(overlay, (x, y), (x + PATCH, y + PATCH), COLORS[cls], -1)

    result = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)

    legend_y = 30
    for i, c in enumerate(CLASSES):
        color = COLORS[i]
        text = f"{c}: {stats[c]}"
        cv2.rectangle(result, (10, legend_y - 18), (25, legend_y + 4), color, -1)
        cv2.putText(result, text, (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        legend_y += 25

    cv2.imwrite(out_path, result)
    print("检测统计:", {k: v for k, v in stats.items() if v > 0})
    print("热力图已保存到:", out_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python predict_aerial.py <航拍图片路径> [输出路径]")
    else:
        out = sys.argv[2] if len(sys.argv) > 2 else "result.jpg"
        predict_aerial(sys.argv[1], out)
