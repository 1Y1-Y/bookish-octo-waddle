import os
import shutil
import random
import sys
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

SRC = r"D:\tobacco\datasets\tobacco_cls"
WORK = r"D:\tobacco\datasets\kfold"
LOG = r"D:\tobacco\runs\kfold_result.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K = 5
EPOCHS = 20
random.seed(42)


def group_key(fn):
    base = os.path.splitext(fn)[0]
    parts = base.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else base


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    print(msg, flush=True)


groups = defaultdict(list)
for cls in os.listdir(SRC):
    d = os.path.join(SRC, cls)
    if not os.path.isdir(d):
        continue
    for f in os.listdir(d):
        groups[group_key(f)].append((cls, f))
group_cls = {gk: Counter(c for c, _ in items).most_common(1)[0][0] for gk, items in groups.items()}

by_cls = defaultdict(list)
for gk, c in group_cls.items():
    by_cls[c].append(gk)
for c in by_cls:
    random.shuffle(by_cls[c])

folds = [[] for _ in range(K)]
for c, gks in by_cls.items():
    for i, gk in enumerate(gks):
        folds[i % K].append(gk)

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def make_fold_dir(fold_idx):
    if os.path.exists(WORK):
        shutil.rmtree(WORK)
    test_gks = set(folds[fold_idx])
    for split in ["train", "test"]:
        os.makedirs(os.path.join(WORK, split), exist_ok=True)
    for gk, items in groups.items():
        split = "test" if gk in test_gks else "train"
        for cls, f in items:
            d = os.path.join(WORK, split, cls)
            os.makedirs(d, exist_ok=True)
            shutil.copy2(os.path.join(SRC, cls, f), os.path.join(d, f))
    return os.path.join(WORK, "train"), os.path.join(WORK, "test")


completed = {}
if os.path.exists(LOG):
    for line in open(LOG, encoding="utf-8"):
        if line.startswith("Fold "):
            idx = int(line.split()[1].split("/")[0]) - 1
            acc = float(line.split("=")[1].split()[0])
            completed[idx] = acc

accs = [completed.get(k) for k in range(K)]
for k in range(K):
    if completed.get(k) is not None:
        continue
    train_dir, test_dir = make_fold_dir(k)
    tr = ImageFolder(train_dir, train_tf)
    te = ImageFolder(test_dir, eval_tf)
    tl = DataLoader(tr, batch_size=32, shuffle=True, num_workers=0)
    el = DataLoader(te, batch_size=32, shuffle=False, num_workers=0)
    ncls = len(tr.classes)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(m.fc.in_features, ncls)
    m = m.to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        m.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(m(x), y).backward()
            opt.step()
        sch.step()
    m.eval()
    tt = tc = 0
    with torch.no_grad():
        for x, y in el:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = m(x)
            tt += y.size(0)
            tc += (out.argmax(1) == y).sum().item()
    acc = tc / tt
    accs[k] = acc
    log(f"Fold {k+1}/{K}: test_acc = {acc:.4f} ({tc}/{tt})")

import statistics
vals = [a for a in accs if a is not None]
if len(vals) == K:
    log(f"K={K}折交叉验证结果: 均值 {statistics.mean(vals):.4f}, 标准差 {statistics.stdev(vals):.4f}")
log(f"各折: {[round(a,4) if a is not None else None for a in accs]}")
