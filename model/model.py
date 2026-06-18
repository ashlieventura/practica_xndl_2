# Laboratori XNDL final notat — classificació d'imatges 32x32 en 14 categories

import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Hiperparàmetres
DATA_DIR    = "../dades"
BATCH_SIZE  = 64
EPOCHS      = 10
LR          = 1e-3
TIME_LIMIT  = 5 * 60  # segons; restricció de l'avaluació

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Les imatges ja són 32x32 i en escala de grisos; normalitzem a [-1, 1]
tf = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

n_classes  = len(train_ds.classes)
input_size = 32 * 32  # cada imatge aplanada és un vector de 1024 valors
print(f"Classes ({n_classes}): {train_ds.classes}")
print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")


# Arquitectura: aplanem la imatge i passem per dues capes lineals
# La capa oculta té 1024 neurones (mateix que la dimensió d'entrada, per ara)
class SimpleFC(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_classes),
        )

    def forward(self, x):
        return self.net(x)


model = SimpleFC(input_size, n_classes).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paràmetres entrenables: {n_params:,}\n")

# CrossEntropyLoss ja inclou el softmax internament
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out       = model(imgs)
            loss_sum += criterion(out, labels).item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
    return correct / total, loss_sum / total


print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>8}  {'Val Acc':>7}")
print("─" * 55)

best_val_acc = 0.0
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    if time.time() - t_start > TIME_LIMIT:
        print(f"Temps exhaurit abans de l'epoch {epoch}. Aturant.")
        break

    model.train()
    correct, total, loss_sum = 0, 0, 0.0

    for imgs, labels in train_loader:
        if time.time() - t_start > TIME_LIMIT:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)

    if total == 0:
        break
    train_acc  = correct / total
    train_loss = loss_sum / total
    val_acc, val_loss = evaluate(val_loader)

    print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
          f"{val_loss:>8.4f}  {val_acc:>6.2%}")

    # guardem el millor model fins ara
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")

print(f"\nMillor val accuracy: {best_val_acc:.2%}")
print(f"Temps total: {time.time()-t_start:.0f}s")
