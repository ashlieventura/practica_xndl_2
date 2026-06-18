# Laboratori XNDL final notat — classificació d'imatges 32x32 en 14 categories
#
# Arquitectura CNN entrenada des de zero (cap pes preentrenat).
# Pensada per a l'entorn d'avaluació d'aquest any: GPU (RTX 3080) i límit de 5 minuts.
#
# Idees clau i per què les fem servir:
#   - CNN profunda tipus VGG (blocs Conv-BN-ReLU x2 + MaxPool): les convolucions
#     detecten patrons locals invariants a la posició, molt més adequades per a
#     imatges que una xarxa fully-connected que ignora l'estructura espacial.
#   - BatchNorm a cada conv: estabilitza i accelera molt la convergència, cosa
#     crítica quan només tenim 5 minuts d'entrenament.
#   - Mixed precision (AMP) + cuDNN benchmark: aprofiten la GPU per fer més
#     èpoques dins del pressupost de temps.
#   - OneCycleLR: programa el learning rate (warmup + descens) per convergir el
#     més ràpid possible; és el que més rendiment dóna amb temps limitat.
#   - Data augmentation moderat (rotacions, flips, translació, escala, erasing):
#     fa el model robust i redueix l'overfitting sense alentir gaire.
#   - label_smoothing: regularització barata que millora la generalització.
#   - Guardem sempre el millor model per val accuracy i autocontrolem el temps.

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- Reproducibilitat ----------
SEED = 123
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# cuDNN en mode benchmark: tria els kernels més ràpids per a mides fixes (32x32)
torch.backends.cudnn.benchmark = True

# ---------- Hiperparàmetres ----------
DATA_DIR    = "../dades"          # ajusta segons l'estructura de l'avaluació
BATCH_SIZE  = 128                 # batch gran: la GPU el suporta i estabilitza BN
LR_MAX      = 3e-3               # pic del OneCycleLR
WEIGHT_DECAY = 5e-4
TIME_LIMIT  = 5 * 60 - 15        # marge de seguretat sobre els 5 min
MAX_EPOCHS  = 40                 # cota alta; el límit real és el temps
LABEL_SMOOTH = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ---------- Transformacions ----------
# Augmentació a train. Nota: treballem en escala de grisos, així que NO usem
# ColorJitter de saturació (no té efecte). Mantenim variacions geomètriques i
# de brillantor/contrast, que sí aporten robustesa.
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.25, contrast=0.25),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
])

# Validació/test: cap augmentació, només normalització coherent amb train.
# IMPORTANT: aquestes mateixes transformacions s'han d'aplicar al test.
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)

# num_workers i pin_memory accel·leren la càrrega de dades cap a la GPU
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=8, pin_memory=True, drop_last=True,
                          persistent_workers=True)
val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

n_classes = len(train_ds.classes)
print(f"Classes ({n_classes}): {train_ds.classes}")
print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")


# ---------- Model ----------
# Bloc bàsic: dues convolucions 3x3 (cada una amb BN+ReLU) seguides de MaxPool.
# Apilar dues convs per bloc augmenta el camp receptiu i la capacitat sense
# disparar el cost, un patró clàssic (estil VGG) que funciona molt bé en 32x32.
def conv_block(c_in, c_out, dropout):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),            # redueix la resolució a la meitat
        nn.Dropout2d(dropout),
    )

class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1,   64,  0.05),   # 32 -> 16
            conv_block(64,  128, 0.10),   # 16 -> 8
            conv_block(128, 256, 0.15),   # 8  -> 4
        )
        # GAP en lloc d'aplanar tot: menys paràmetres, menys overfitting i
        # independent de petites variacions de mida.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


model = CNN(n_classes).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Paràmetres entrenables: {n_params:,}\n")

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

# OneCycleLR: necessita saber el nombre total de passos. Estimem quantes èpoques
# caben en el temps fent una primera època cronometrada i recalibrant; per
# simplicitat fixem l'scheduler sobre MAX_EPOCHS i aturem per temps. Funciona bé
# perquè la fase de warmup és curta i el LR baixa progressivament.
steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR_MAX,
    steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
    pct_start=0.15, div_factor=10, final_div_factor=100,
)

# AMP: càlcul en mitja precisió on és segur -> més ràpid a la GPU
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
        # OneCycle avança per cada batch; protegim contra passos extra
        if scheduler.last_epoch < scheduler.total_steps - 1:
            scheduler.step()

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
        torch.save(model.state_dict(), "best_model.pt")

print(f"\nMillor val accuracy: {best_val_acc:.2%}")
print(f"Temps total: {time.time()-t_start:.0f}s")

# Per provar el millor model:
#   model = CNN(n_classes).to(device)
#   model.load_state_dict(torch.load("best_model.pt", map_location=device))
#   model.eval()
