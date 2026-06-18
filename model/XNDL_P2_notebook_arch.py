# Laboratori XNDL final notat — classificació d'imatges 32x32 en 14 categories
#
# Aquesta versió adapta l'arquitectura del notebook de Quick, Draw! de David
# Kradolfer (2017), que al seu torn segueix el model CNN del tutorial de Jason
# Brownlee per a reconeixement de dígits amb Keras. L'estructura original és:
#
#   Conv(30, 5x5) -> MaxPool(2x2) -> Conv(15, 3x3) -> MaxPool(2x2)
#   -> Dropout(0.2) -> Flatten -> Dense(128) -> Dense(50) -> softmax
#
# La traduïm a PyTorch i l'adaptem al nostre cas (entrades 32x32 en escala de
# grisos i 14 classes en lloc de 28x28 i 2/5 classes). Mantenim l'esperit del
# model original però afegim els elements imprescindibles per a un entrenament
# competitiu dins de les restriccions d'aquest any (GPU + 5 min):
#   - BatchNorm: el model original no en tenia; aquí accelera molt la convergència.
#   - Data augmentation lleuger: el notebook no en feia servir, però per a només
#     5 minuts d'entrenament ajuda a generalitzar sense gaire cost.
# Aquests dos canvis es comenten i es poden desactivar si es vol reproduir
# l'arquitectura "pura" del notebook.

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- Reproducibilitat ----------
SEED = 0  # el notebook original feia servir random_state=0
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

# ---------- Hiperparàmetres ----------
DATA_DIR   = "../dades"
BATCH_SIZE = 200          # mateix batch_size que al notebook (model.fit(..., batch_size=200))
LR         = 1e-3         # Adam, com a l'original ('adam')
TIME_LIMIT = 5 * 60 - 15
MAX_EPOCHS = 40           # el notebook entrenava 10 èpoques; aquí limitem per temps
USE_BN     = True         # afegit nostre (no era a l'original)
USE_AUG    = True         # afegit nostre (no era a l'original)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ---------- Transformacions ----------
if USE_AUG:
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
else:
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

n_classes = len(train_ds.classes)
print(f"Classes ({n_classes}): {train_ds.classes}")
print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")


# ---------- Model (arquitectura del notebook, adaptada) ----------
# Notebook: Conv(30,5x5) sense padding sobre 28x28 -> 24x24, pool -> 12x12,
#           Conv(15,3x3) -> 10x10, pool -> 5x5, dropout, flatten, 128, 50, out.
# Aquí l'entrada és 32x32. Mantenim els mateixos kernels i sense padding perquè
# el reduït espacial sigui anàleg: 32 -conv5-> 28 -pool-> 14 -conv3-> 12 -pool-> 6.
class NotebookCNN(nn.Module):
    def __init__(self, n_classes, use_bn=True):
        super().__init__()
        def maybe_bn(c):
            return nn.BatchNorm2d(c) if use_bn else nn.Identity()

        self.features = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=5),   # 32 -> 28
            maybe_bn(30),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 28 -> 14

            nn.Conv2d(30, 15, kernel_size=3),  # 14 -> 12
            maybe_bn(15),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                   # 12 -> 6

            nn.Dropout2d(0.2),                 # Dropout 20% com a l'original
        )
        # després dels blocs: 15 canals x 6 x 6 = 540
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(15 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = NotebookCNN(n_classes, use_bn=USE_BN).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paràmetres entrenables: {n_params:,}\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
use_amp = (device.type == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)
    return correct / total, loss_sum / total


print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}  {'t(s)':>6}")
print("-" * 60)

best_val_acc = 0.0
t_start = time.time()
stop = False

for epoch in range(1, MAX_EPOCHS + 1):
    if stop or time.time() - t_start > TIME_LIMIT:
        break
    model.train()
    correct, total, loss_sum = 0, 0, 0.0
    for imgs, labels in train_loader:
        if time.time() - t_start > TIME_LIMIT:
            stop = True
            break
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)
    if total == 0:
        break
    train_acc, train_loss = correct / total, loss_sum / total
    val_acc, val_loss = evaluate(val_loader)
    elapsed = time.time() - t_start
    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
          f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>6.0f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model_notebook.pt")

print(f"\nMillor val accuracy: {best_val_acc:.2%}")
print(f"Temps total: {time.time()-t_start:.0f}s")
