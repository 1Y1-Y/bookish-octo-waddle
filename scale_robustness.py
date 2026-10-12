"""尺度鲁棒性测试：模拟无人机不同高度（目标变小）时模型的识别能力。

原理：航拍时烟叶在画面中占比很小。把测试图缩小后贴到大画布上（周围模拟田间背景），
再送模型，即可模拟"目标变小"对识别的影响。
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8")

CLASSES = [
    "Black Spot", "Black shank", "Healthy", "Leaf Spot",
    "Tobacco leaf curl disease", "Tobacco mosaic virus", "Yellow Leaf Curl Virus", "brown sport",
]
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else r"D:\tobacco\runs\cls_model.pt"
DATA = r"D:\tobacco\datasets\tobacco_cls_group\test"
CANVAS = 640
SCALES = [1.0, 0.5, 0.25, 0.125]

TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model():
    m = torchvision.models.mobilenet_v3_small()
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return m.cuda().eval()


def make_canvas(img, scale):
    """把图按比例缩放后贴到田间背景大画布中央。
    scale=1.0 时铺满画布（等价于直接识别）；scale 越小表示目标在画面中越小（越高）。"""
    h, w = img.shape[:2]
    s = (CANVAS * scale) / max(h, w)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    small = cv2.resize(img, (nw, nh))
    canvas = np.full((CANVAS, CANVAS, 3), (60, 80, 50), dtype=np.uint8)  # 田间背景色(暗绿)
    y0 = (CANVAS - nh) // 2
    x0 = (CANVAS - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas


def main():
    model = load_model()
    # 收集测试图
    samples = []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(DATA, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d)[:40]:
            samples.append((os.path.join(d, f), ci))

    print(f"测试样本数: {len(samples)}")
    print(f"{'缩放比例':<12}{'对应飞行高度示意':<18}{'准确率'}")
    print("-" * 46)
    for scale in SCALES:
        correct = 0
        for path, gt in samples:
            img = cv2.imread(path)
            if img is None:
                continue
            canvas = make_canvas(img, scale)
            t = TF(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
            with torch.no_grad():
                pred = model(t).argmax(1).item()
            correct += int(pred == gt)
        acc = correct / len(samples)
        if scale == 1.0:
            note = "近景（原始）"
        elif scale == 0.5:
            note = "约 2 倍高度"
        elif scale == 0.25:
            note = "约 4 倍高度"
        else:
            note = "约 8 倍高度"
        print(f"{scale:<12}{note:<18}{acc:.3f}")


if __name__ == "__main__":
    main()
