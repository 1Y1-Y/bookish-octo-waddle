import argparse
import os
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(name, num_classes):
    if name == "resnet18":
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "resnet50":
        m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "mobilenetv3":
        m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def num_params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imbalance", default="none", choices=["none", "weighted", "oversample", "focal"])
    ap.add_argument("--tag", default="exp")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

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

    train_ds = ImageFolder(os.path.join(args.data, "train"), train_tf)
    val_ds = ImageFolder(os.path.join(args.data, "val"), eval_tf)
    test_ds = ImageFolder(os.path.join(args.data, "test"), eval_tf)
    ncls = len(train_ds.classes)

    sampler = None
    if args.imbalance == "oversample":
        cnt = Counter([s for _, s in train_ds.samples])
        w = [1.0 / cnt[s] for _, s in train_ds.samples]
        sampler = WeightedRandomSampler(w, len(w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = build_model(args.model, ncls).to(DEVICE)
    nparam = num_params(model)

    if args.imbalance == "weighted":
        cnt = Counter([s for _, s in train_ds.samples])
        w = torch.tensor([1.0 / max(cnt.get(i, 1), 1) for i in range(ncls)])
        w = w / w.sum() * ncls
        criterion = nn.CrossEntropyLoss(weight=w.to(DEVICE))
    elif args.imbalance == "focal":
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        vt = vc = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                vt += labels.size(0)
                vc += (out.argmax(1) == labels).sum().item()
        va = vc / vt
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), f"D:/tobacco/runs/{args.tag}_best.pt")

    model.load_state_dict(torch.load(f"D:/tobacco/runs/{args.tag}_best.pt"))
    model.eval()
    tt = tc = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            tt += labels.size(0)
            tc += (out.argmax(1) == labels).sum().item()
    print(f"RESULT tag={args.tag} model={args.model} imbalance={args.imbalance} "
          f"params={nparam:.1f}M best_val={best_acc:.4f} test_acc={tc/tt:.4f} ({tc}/{tt})")


if __name__ == "__main__":
    main()
