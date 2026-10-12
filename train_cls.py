import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader
import os

DATA = r"D:\tobacco\datasets\tobacco_cls_split"
NUM_CLASSES = 8
BATCH = 32
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

train_ds = ImageFolder(os.path.join(DATA, "train"), train_tf)
val_ds = ImageFolder(os.path.join(DATA, "val"), eval_tf)
test_ds = ImageFolder(os.path.join(DATA, "test"), eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

class_names = train_ds.classes
print("类别:", class_names)
print("数据量: train", len(train_ds), "val", len(val_ds), "test", len(test_ds))

model = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

os.makedirs(r"D:\tobacco\runs", exist_ok=True)
best_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    total = correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total += labels.size(0)
        correct += (out.argmax(1) == labels).sum().item()
    train_acc = correct / total

    model.eval()
    vtotal = vcorrect = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            vtotal += labels.size(0)
            vcorrect += (out.argmax(1) == labels).sum().item()
    val_acc = vcorrect / vtotal
    scheduler.step()
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), r"D:\tobacco\runs\cls_best.pt")
    print(f"epoch {epoch+1}/{EPOCHS} | train_acc {train_acc:.4f} | val_acc {val_acc:.4f}")

print(f"最佳 val_acc: {best_acc:.4f}")

model.load_state_dict(torch.load(r"D:\tobacco\runs\cls_best.pt"))
model.eval()
ttotal = tcorrect = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        ttotal += labels.size(0)
        tcorrect += (out.argmax(1) == labels).sum().item()
print(f"测试集准确率: {tcorrect/ttotal:.4f} ({tcorrect}/{ttotal})")
