# cnn_resnet9_att.py — ResNet-9 + atenció CBAM per a classificació 32×32
#
# Per què aquests dos canvis respecte al model anterior (85.75% val acc)?
#
# 1. RESNET-9: el model anterior era VGG-style (cap connexió de skip).
#    Quan l'acc s'estanca i el train acc segueix pujant, sol indicar que
#    la xarxa té prou capacitat però els gradients no flueixen bé cap a
#    les primeres capes (vanishing gradient). Les connexions residuals
#    (x + F(x)) permeten que el gradient salti directament, facilitant
#    que les capes inicials continuïn aprenent fins al final.
#    ResNet-9 és la versió mínima: 4 blocs conv + 2 skip connections,
#    molt ràpida d'entrenar i habitualment millor que VGG a 32×32.
#
# 2. ATENCIÓ CBAM (Channel + Spatial): les classes es distingeixen pels
#    DETALLS INTERNS dels cercles. CBAM afegeix dos mòduls lleugers:
#    - Channel attention: "quins canals (detectors de trets) importa
#      més activar per a aquesta imatge?" → pes per canal.
#    - Spatial attention: "on de la imatge cal fixar-se?" → pes per píxel.
#    Això guia el model a centrar-se en l'interior de l'objecte rodó,
#    que és exactament el que discrimina les classes entre elles.
#    Cost: ~0 paràmetres addicionals rellevants, però sí millora notable.

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR    = os.getenv("DATA_DIR", os.path.join(PROJECT_DIR, "dades"))

# ================================================================
# CONFIG
# ================================================================
CONFIG = {
    "data_dir":    DATA_DIR,
    "batch_size":  512,

    # Augmentació: erasing_medium, el millor del grid search
    "augment":     "erasing_medium",

    # Entrenament
    "lr":           1e-3,
    # Baixem lleugerament el lr respecte al model anterior (2e-3):
    # ResNet-9 amb skip connections és més estable i no necessita
    # un lr tan agressiu per escapar mínims dolents; un valor més
    # conservador ajuda a afinar millor els últims epochs.
    "optimizer":    "adamw",
    # AdamW en comptes d'Adam: desacobla el weight decay dels moments
    # adaptatius. Amb ResNet i skip connections, el weight decay ben
    # aplicat és important per evitar que els pesos creixin sense control.
    "weight_decay": 1e-4,
    "scheduler":    "onecycle",
    "reducelr_factor":   0.5,
    "reducelr_patience": 5,
    "label_smooth": 0.05,
    # Label smoothing baix (0.05): amb 14 classes bastant diferenciades
    # visualment, no volem suavitzar massa els targets — el model ha
    # d'aprendre a ser prou confident en les diferències subtils.
    "max_epochs":   55,
    # Més epochs que l'anterior (55→60) perquè ResNet-9 és més ràpida
    # per epoch que el VGG-style (menys paràmetres, operacions més
    # eficients) i amb una GPU més potent que una 3080 podem aprofitar-ho.
    "time_limit":   40 * 60 - 15,

    "seed":         42,
    "cache_in_ram": True,
}
# ================================================================


# ================================================================
# ATENCIÓ CBAM
# ================================================================

class ChannelAttention(nn.Module):
    """Atenció per canal: aprèn a ponderar quins mapes de característiques
    (canals) són rellevants per a la classificació actual.
    Usa tant AvgPool com MaxPool per capturar presència mitjana i màxima
    del tret — combinar-les dóna millors resultats que usar-ne una sola.
    El ratio de reducció (r=8) controla el coll d'ampolla intern:
    prou expressiu però quasi sense cost computacional."""
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        avg = self.fc(x.mean(dim=[2, 3]))          # [B, C]
        mx  = self.fc(x.amax(dim=[2, 3]))          # [B, C]
        w   = torch.sigmoid(avg + mx).unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
        return x * w


class SpatialAttention(nn.Module):
    """Atenció espacial: aprèn a ponderar ON de la imatge cal fixar-se.
    Clau per al nostre problema: els objectes rodons es distingeixen
    per l'interior, no per la silueta. L'atenció espacial pot aprendre
    a ignorar el contorn circular (igual a totes les classes) i
    concentrar-se en la textura/patró central.
    Kernel 7×7 per tenir camp receptiu gran sobre imatges 4×4 (post-pool)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)    # [B, 1, H, W]
        mx  = x.amax(dim=1, keepdim=True)    # [B, 1, H, W]
        w   = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """CBAM = Channel Attention → Spatial Attention, en sèrie.
    S'aplica com a mòdul plug-in al final de cada bloc residual."""
    def __init__(self, channels, r=8):
        super().__init__()
        self.ca = ChannelAttention(channels, r)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


# ================================================================
# ARQUITECTURA RESNET-9
# ================================================================

def conv_bn(c_in, c_out, kernel=3, stride=1, padding=1):
    """Bloc bàsic: Conv → BN → ReLU. Bias=False perquè BN ja té beta."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ResNet9(nn.Module):
    """ResNet-9 adaptat per a imatges 32×32 en escala de grisos amb CBAM.

    Estructura:
      prep  : Conv 1→64,  32×32
      layer1: Conv 64→128, 32×32 → MaxPool → 16×16  +  residual (skip)
      layer2: Conv 128→256, 16×16 → MaxPool → 8×8   (sense skip, bloc simple)
      layer3: Conv 256→512, 8×8  → MaxPool → 4×4    +  residual (skip)
      CBAM  : atenció canal + espacial sobre els 512 mapes finals
      head  : MaxPool → Flatten → Dropout → FC → 14 classes

    Les skip connections (residuals) permeten que els gradients flueixin
    directament fins a les primeres capes, evitant l'estancament que
    patia el model VGG-style anterior als ~85%.
    """
    def __init__(self, n_classes, dropout=0.3):
        super().__init__()

        # Preparació: extreu trets bàsics sense reduir resolució
        self.prep = conv_bn(1, 64, kernel=3, stride=1, padding=1)   # 32×32

        # Bloc 1 amb skip: duplica canals i fa downsampling
        self.layer1      = conv_bn(64, 128, stride=1)                # 32×32
        self.layer1_pool = nn.MaxPool2d(2)                           # 16×16
        # Connexió skip: iguala canals per poder sumar (1×1 conv, sense padding)
        self.res1 = nn.Sequential(
            conv_bn(128, 128, stride=1),
            conv_bn(128, 128, stride=1),
        )

        # Bloc 2 (sense skip): augmenta capacitat de representació
        self.layer2      = conv_bn(128, 256, stride=1)               # 16×16
        self.layer2_pool = nn.MaxPool2d(2)                           # 8×8

        # Bloc 3 amb skip: arriba a 512 canals
        self.layer3      = conv_bn(256, 512, stride=1)               # 8×8
        self.layer3_pool = nn.MaxPool2d(2)                           # 4×4
        self.res3 = nn.Sequential(
            conv_bn(512, 512, stride=1),
            conv_bn(512, 512, stride=1),
        )

        # CBAM: guia l'atenció cap als detalls interns de l'objecte rodó
        self.cbam = CBAM(512, r=8)

        # Cap de classificació
        self.pool    = nn.MaxPool2d(4)                               # 4×4 → 1×1
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.prep(x)                              # 1×32×32 → 64×32×32

        x = self.layer1(x)                            # 64×32×32 → 128×32×32
        x = self.layer1_pool(x)                       # 128×32×32 → 128×16×16
        x = x + self.res1(x)                          # skip connection

        x = self.layer2(x)                            # 128×16×16 → 256×16×16
        x = self.layer2_pool(x)                       # 256×16×16 → 256×8×8

        x = self.layer3(x)                            # 256×8×8 → 512×8×8
        x = self.layer3_pool(x)                       # 512×8×8 → 512×4×4
        x = x + self.res3(x)                          # skip connection

        x = self.cbam(x)                              # atenció canal + espacial

        x = self.pool(x).flatten(1)                   # 512×4×4 → 512×1×1 → 512
        x = self.dropout(x)
        return self.fc(x)


# ================================================================
# DADES
# ================================================================

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        base           = datasets.ImageFolder(folder)
        self.transform = transform
        self.targets   = [s[1] for s in base.samples]
        self.classes   = base.classes
        cache          = folder.rstrip("/\\") + "_imgs.npy"
        if os.path.exists(cache):
            print(f"  Caché trobada, carregant...", end=" ", flush=True)
            t0 = time.time()
            self._imgs = np.load(cache)
            print(f"fet en {time.time()-t0:.1f}s  ({self._imgs.nbytes//1024//1024}MB)")
        else:
            print(f"  Primera càrrega: {len(base.samples):,} imatges...")
            paths = [p for p, _ in base.samples]
            def _read(p): return np.array(PILImage.open(p).convert("L"))
            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(tqdm(ex.map(_read, paths), total=len(paths), unit="img"))
            self._imgs = np.stack(imgs); del imgs
            np.save(cache, self._imgs)
            print(f"  Caché guardada → {cache}")

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        img = PILImage.fromarray(self._imgs[idx], mode="L")
        if self.transform: img = self.transform(img)
        return img, self.targets[idx]


def make_transforms(augment):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if augment == "none":
        return transforms.Compose(base), transforms.Compose(base)
    elif augment == "light":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif augment == "erasing_medium":
        # El millor del grid search: rotació + translació + erasing zones mitjanes.
        # Força el model a reconèixer l'objecte encara que part del centre estigui esborrat.
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(15),
            # Rotem una mica més que abans (10→15) perquè els objectes
            # són circulars i la rotació no canvia la semàntica.
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.5, scale=(0.08, 0.20), ratio=(0.5, 2.0), value=0),
        ])
    else:
        raise ValueError(f"augment desconegut: {augment!r}")
    return train_tf, transforms.Compose(base)


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(cfg, device):
    train_tf, eval_tf = make_transforms(cfg["augment"])
    DS = InMemoryDataset if cfg.get("cache_in_ram") else datasets.ImageFolder
    train_ds = DS(os.path.join(cfg["data_dir"], "train"), transform=train_tf)
    val_ds   = DS(os.path.join(cfg["data_dir"], "val"),   transform=eval_tf)
    pin = (device.type == "cuda")
    nw  = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


# ================================================================
# OPTIMITZADOR I SCHEDULER
# ================================================================

def build_optimizer(cfg, model):
    lr, wd   = cfg["lr"], cfg["weight_decay"]
    decay    = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups   = [{"params": decay,    "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]
    if cfg["optimizer"] == "adamw": return optim.AdamW(groups, lr=lr)
    if cfg["optimizer"] == "adam":  return optim.Adam(groups, lr=lr)
    if cfg["optimizer"] == "sgd":   return optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(cfg["optimizer"])


def build_scheduler(cfg, optimizer, steps_per_epoch):
    if cfg["scheduler"] == "none":     return None
    if cfg["scheduler"] == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"],
            steps_per_epoch=steps_per_epoch, epochs=cfg["max_epochs"],
            pct_start=0.10, div_factor=10, final_div_factor=100,
        )
    if cfg["scheduler"] == "reducelr":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg["reducelr_factor"], patience=cfg["reducelr_patience"],
        )
    if cfg["scheduler"] == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    raise ValueError(cfg["scheduler"])


# ================================================================
# AVALUACIÓ
# ================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out  = model(imgs)
            loss = criterion(out, labels)
        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)
    return correct / total, loss_sum / total


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device}\n")

    train_loader, val_loader, classes = make_loaders(CONFIG, device)
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")
    print(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}\n")

    model    = ResNet9(n_classes, dropout=0.3).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {n_params:,}")
    print(model)
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smooth"])
    optimizer = build_optimizer(CONFIG, model)
    scheduler = build_scheduler(CONFIG, optimizer, len(train_loader))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'t(s)':>6}  {'LR':>8}")
    print("-" * 70)

    best_val_acc = 0.0
    t_start      = time.time()
    stop         = False

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        if stop or time.time() - t_start > CONFIG["time_limit"]:
            break

        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Època {epoch:>2}", leave=False,
                    unit="batch", dynamic_ncols=True)
        for imgs, labels in pbar:
            if time.time() - t_start > CONFIG["time_limit"]:
                stop = True; break
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and CONFIG["scheduler"] == "onecycle":
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()

            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        if total == 0: break

        train_acc, train_loss = correct / total, loss_sum / total
        val_acc, val_loss     = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed               = time.time() - t_start

        if scheduler is not None:
            if CONFIG["scheduler"] == "reducelr":
                scheduler.step(val_loss)
            elif CONFIG["scheduler"] == "step":
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.0f}s  lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMillor val accuracy: {best_val_acc:.2%}")
    print(f"Temps total: {time.time() - t_start:.0f}s")
