# Laboratori XNDL final notat — classificació d'imatges 32x32 en 14 categories
#
# Canvis respecte la versió anterior:
#   - num_workers reduït a 2 (Windows spawn té molt overhead amb workers alts)
#   - Augmentació pesada moguda a GPU amb transforms.v2 / kornia-style inline
#   - prefetch_factor per solapar càrrega i còmput
#   - TIME_LIMIT corregit per aturar dins del bucle de batches

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

torch.backends.cudnn.benchmark = True

# ---------- Hiperparàmetres ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "dades", "dades")
BATCH_SIZE   = 256    # augmentem batch: GPU infrautilitzada, posem-li més feina
LR_MAX       = 3e-3
WEIGHT_DECAY = 5e-4
TIME_LIMIT   = 5 * 60 - 15
MAX_EPOCHS   = 40
LABEL_SMOOTH = 0.05

# ---------- Transformacions ----------
# Augmentació MÍNIMA en CPU (només les que no tenen alternativa GPU fàcil).
# La resta es fa inline al bucle sobre tensors ja a CUDA → molt més ràpid.
train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ---------- Augmentació GPU (inline al bucle) ----------
# Apliquem rotació + translació + erasing directament sobre el batch a CUDA.
# Molt més ràpid que fer-ho per imatge en CPU amb múltiples workers.
def gpu_augment(imgs):
    """imgs: (B,1,H,W) float tensor ja a CUDA, rang [-1,1]"""
    B, C, H, W = imgs.shape

    # Random horizontal flip (ja fet a CPU, però per si de cas)
    # Random rotation ±20° via affine grid
    angle = (torch.rand(B, device=imgs.device) * 40 - 20) * (3.14159 / 180)
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    zeros = torch.zeros(B, device=imgs.device)
    ones  = torch.ones(B,  device=imgs.device)
    # Matriu affine 2x3 per rotació
    theta = torch.stack([
        torch.stack([cos_a, -sin_a, zeros], dim=1),
        torch.stack([sin_a,  cos_a, zeros], dim=1),
    ], dim=1)  # (B, 2, 3)
    grid = torch.nn.functional.affine_grid(theta, imgs.size(), align_corners=False)
    imgs = torch.nn.functional.grid_sample(imgs, grid, align_corners=False, padding_mode='reflection')

    # Random translation ±10%
    tx = (torch.rand(B, device=imgs.device) * 0.2 - 0.1)
    ty = (torch.rand(B, device=imgs.device) * 0.2 - 0.1)
    theta_t = torch.stack([
        torch.stack([ones,  zeros, tx], dim=1),
        torch.stack([zeros, ones,  ty], dim=1),
    ], dim=1)
    grid_t = torch.nn.functional.affine_grid(theta_t, imgs.size(), align_corners=False)
    imgs = torch.nn.functional.grid_sample(imgs, grid_t, align_corners=False, padding_mode='reflection')

    # Brightness/contrast jitter (escala de grisos: simple multiplicació + offset)
    brightness = torch.rand(B, 1, 1, 1, device=imgs.device) * 0.5 + 0.75  # [0.75, 1.25]
    contrast   = torch.rand(B, 1, 1, 1, device=imgs.device) * 0.5 + 0.75
    mean = imgs.mean(dim=[2, 3], keepdim=True)
    imgs = (imgs - mean) * contrast + mean * brightness
    imgs = imgs.clamp(-1, 1)

    # Random erasing (~20% dels exemples)
    mask = torch.rand(B, device=imgs.device) < 0.2
    if mask.any():
        h_e = int(H * 0.25)
        w_e = int(W * 0.25)
        for i in mask.nonzero(as_tuple=True)[0]:
            y0 = torch.randint(0, H - h_e, (1,)).item()
            x0 = torch.randint(0, W - w_e, (1,)).item()
            imgs[i, :, y0:y0+h_e, x0:x0+w_e] = 0.0

    return imgs

# ---------- Model ----------
def conv_block(c_in, c_out, dropout):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
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
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)

    # Windows: num_workers=2 és el punt dolç. Més workers → overhead spawn > guany
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    n_classes = len(train_ds.classes)
    print(f"Classes ({n_classes}): {train_ds.classes}")
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

    model = CNN(n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {n_params:,}\n")

    # Diagnòstic ràpid: la GPU respon?
    if device.type == "cuda":
        x = torch.randn(2048, 2048, device=device)
        t0 = time.time()
        for _ in range(50):
            _ = x @ x
        torch.cuda.synchronize()
        print(f"[GPU benchmark] 50x matmul 2048x2048: {time.time()-t0:.2f}s  ← hauria de ser <2s\n")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_MAX,
        steps_per_epoch=steps_per_epoch, epochs=MAX_EPOCHS,
        pct_start=0.15, div_factor=10, final_div_factor=100,
    )

    use_amp = (device.type == "cuda")
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    @torch.no_grad()
    def evaluate(loader):
        model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(imgs)
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

            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Augmentació a GPU (ràpid, sense cost de CPU addicional)
            imgs = gpu_augment(imgs)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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