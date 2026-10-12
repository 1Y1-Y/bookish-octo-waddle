"""多尺度增强训练：模拟航拍中"目标大小随机变化"的场景，提升模型尺度鲁棒性。"""
import os
import sys
import random
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from PIL import Image

DATA = r"D:\tobacco\datasets\tobacco_cls_group"
OUT = r"D:\tobacco\runs\cls_model_scale.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH = 32


class RandomScalePad:
    """随机把图片缩放后贴到田间背景画布上，模拟目标大小/距离变化"""

    def __init__(self, size=224, bg=(60, 80, 50)):
        self.size = size
        self.bg = bg

    def __call__(self, img):
        w, h = img.size
        s = random.uniform(0.4, 1.0)
        scale = (self.size * s) / max(w, h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        img = img.resize((nw, nh))
        canvas = Image.new("RGB", (self.size, self.size), self.bg)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


train_tf = transforms.Compose([
    RandomScalePad(224),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = ImageFolder(os.path.join(DATA, "train"), train_tf)
val_ds = ImageFolder(os.path.join(DATA, "val"), eval_tf)
test_ds = ImageFolder(os.path.join(DATA, "test"), eval_tf)

cnt = Counter([s for _, s in train_ds.samples])
w = [1.0 / cnt[s] for _, s in train_ds.samples]
sampler = WeightedRandomSampler(w, len(w), replacement=True)

tl = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, num_workers=0)
vl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
el = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(train_ds.classes))
m = m.to(DEVICE)
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(m.parameters(), lr=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best = 0.0
for ep in range(EPOCHS):
    m.train()
    for x, y in tl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        crit(m(x), y).backward()
        opt.step()
    sch.step()
    m.eval()
    vt = vc = 0
    with torch.no_grad():
        for x, y in vl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            o = m(x)
            vt += y.size(0)
            vc += (o.argmax(1) == y).sum().item()
    if vc / vt > best:
        best = vc / vt
        torch.save(m.state_dict(), OUT)
    print(f"epoch {ep+1}/{EPOCHS} val_acc {vc/vt:.4f}", flush=True)

m.load_state_dict(torch.load(OUT))
m.eval()
tt = tc = 0
with torch.no_grad():
    for x, y in el:
        x, y = x.to(DEVICE), y.to(DEVICE)
        o = m(x)
        tt += y.size(0)
        tc += (o.argmax(1) == y).sum().item()
print(f"RESULT scale-augmented model: best_val={best:.4f} test_acc={tc/tt:.4f} ({tc}/{tt})")
