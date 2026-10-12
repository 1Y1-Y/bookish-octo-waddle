"""鲁棒性测试：给测试图施加真实场景干扰（模糊/光照/噪声/旋转/压缩），看模型准确率下降多少。"""
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

TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path):
    m = torchvision.models.mobilenet_v3_small()
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.cuda().eval()


def deg_blur(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)


def deg_bright(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def deg_noise(img, sigma):
    n = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def deg_rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def deg_jpeg(img, q):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def eval_degradation(model, samples, fn):
    correct = 0
    for path, gt in samples:
        img = cv2.imread(path)
        if img is None:
            continue
        img = fn(img)
        t = TF(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
        with torch.no_grad():
            pred = model(t).argmax(1).item()
        correct += int(pred == gt)
    return correct / len(samples)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else r"D:\tobacco\runs\cls_model.pt"
    model = load_model(model_path)
    DATA = r"D:\tobacco\datasets\tobacco_cls_group\test"
    samples = []
    for ci, cls in enumerate(CLASSES):
        d = os.path.join(DATA, cls)
        if os.path.isdir(d):
            for f in os.listdir(d)[:40]:
                samples.append((os.path.join(d, f), ci))

    print(f"模型: {os.path.basename(model_path)}  样本数: {len(samples)}")
    print(f"{'干扰类型':<20}{'准确率'}")
    print("-" * 32)
    print(f"{'原始（无干扰）':<18}{eval_degradation(model, samples, lambda x: x):.3f}")
    print(f"{'轻微模糊':<19}{eval_degradation(model, samples, lambda x: deg_blur(x, 5)):.3f}")
    print(f"{'严重模糊':<19}{eval_degradation(model, samples, lambda x: deg_blur(x, 11)):.3f}")
    print(f"{'暗光(0.5x)':<18}{eval_degradation(model, samples, lambda x: deg_bright(x, 0.5)):.3f}")
    print(f"{'暗光(0.3x)':<18}{eval_degradation(model, samples, lambda x: deg_bright(x, 0.3)):.3f}")
    print(f"{'过曝(1.6x)':<18}{eval_degradation(model, samples, lambda x: deg_bright(x, 1.6)):.3f}")
    print(f"{'噪点(sigma=20)':<16}{eval_degradation(model, samples, lambda x: deg_noise(x, 20)):.3f}")
    print(f"{'旋转15度':<18}{eval_degradation(model, samples, lambda x: deg_rotate(x, 15)):.3f}")
    print(f"{'旋转30度':<18}{eval_degradation(model, samples, lambda x: deg_rotate(x, 30)):.3f}")
    print(f"{'JPEG低质量(30)':<15}{eval_degradation(model, samples, lambda x: deg_jpeg(x, 30)):.3f}")


if __name__ == "__main__":
    main()
