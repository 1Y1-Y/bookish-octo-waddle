import cv2
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import numpy as np
import sys
from collections import Counter

from disease_knowledge import KNOWLEDGE, severity_level

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

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cls_model.pt")
PATCH = 224
STRIDE = 112
CONF = 0.5


def load_model():
    model = torchvision.models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    return model.cuda().eval()


TF = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def scan_image(model, img):
    h, w = img.shape[:2]
    counts = Counter()
    total = 0
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
            total += 1
            counts[cls] += 1
    return counts, total


def generate_report(image_path):
    model = load_model()
    img = cv2.imread(image_path)
    if img is None:
        return "无法读取图片: " + image_path

    counts, total = scan_image(model, img)
    if total == 0:
        return "未在图像中检测到明显的病害区域（或图片质量过低）。"

    diseases = {CLASSES[i]: n for i, n in counts.items() if CLASSES[i] != "Healthy"}
    healthy_n = counts.get(CLASSES.index("Healthy"), 0)

    lines = []
    lines.append("=" * 40)
    lines.append("        烟田病害诊断与施药建议报告")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"【一、病情总览】")
    lines.append(f"共分析 {total} 个区块，其中健康区块 {healthy_n} 个，发病区块 {total - healthy_n} 个。")
    lines.append("")

    sorted_d = sorted(diseases.items(), key=lambda kv: -kv[1])
    lines.append("【二、病害分析与施药建议】")
    if not sorted_d:
        lines.append("未发现病害，烟田总体健康。")
    for name, n in sorted_d:
        ratio = n / total
        level = severity_level(ratio)
        k = KNOWLEDGE[name]
        lines.append("")
        lines.append(f"◆ {k['中文名']}（{name}）—— 严重程度：{level}（{n} 个区块，占 {ratio:.0%}）")
        lines.append(f"  病原：{k['病原']}")
        lines.append(f"  症状：{k['症状']}")
        lines.append(f"  用药建议：{k['防治药剂']}")
        lines.append(f"  预防措施：{k['预防措施']}")
    lines.append("")

    lines.append("【三、综合预防与风险提示】")
    viral = [d for d in sorted_d if "病毒" in KNOWLEDGE[d[0]]["病原"]]
    fungal = [d for d in sorted_d if "真菌" in KNOWLEDGE[d[0]]["病原"]]
    bacterial = [d for d in sorted_d if "细菌" in KNOWLEDGE[d[0]]["病原"]]
    if viral:
        lines.append(f"· 发现 {len(viral)} 种病毒病，注意防治传毒媒介（烟粉虱、蚜虫），及时清除病株，防止扩散。")
    if fungal:
        lines.append(f"· 发现 {len(fungal)} 种真菌病，注意雨后及时排水排湿、清除病残叶，防止再次侵染。")
    if bacterial:
        lines.append("· 发现细菌性病害（黑胫病），注意轮作和土壤处理，病株要拔除销毁，避免连作加重。")
    if healthy_n / total < 0.5:
        lines.append("· 健康区块占比偏低，病情扩散风险高，建议对病害集中区域优先施药，并加强田间巡查。")
    lines.append("")
    lines.append("【四、用药安全提醒】")
    lines.append("· 施药时遵守农药安全使用规范，严格按说明书浓度和间隔期操作，注意安全防护。")
    lines.append("· 采收前注意农药安全间隔期，确保烟叶农残达标。")
    lines.append("")

    report = "\n".join(lines)
    with open("diagnosis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python agent.py <航拍图片路径>")
    else:
        print(generate_report(sys.argv[1]))
        print("\n[报告已保存到 diagnosis_report.txt]")
