# cnn_v2.py — CNN millorada per a classificació d'imatges 32×32 en 14 categories
#
# Canvis respecte a la versió anterior (cnn_pooling_raro.py):
#
#  1. MaxPool en lloc de Conv stride=2: la conv amb stride aprèn el downsampling
#     però té més paràmetres i sobreajusta més. MaxPool és invariant a petites
#     traslacions i generalitza millor en datasets petits.
#
#  2. Data augmentation: imatges com lluna, sol, roda, cercle... són simètrics
#     o quasi-simètrics, per tant flip horitzontal i rotació petita ajuden molt.
#     Afegim també petites traslacions (RandomAffine) per fer el model robust
#     a la posició de l'objecte.
#
#  3. Global Average Pooling (GAP) abans del cap FC: en lloc d'aplanar tot el
#     feature map (que pot ser gran), fem la mitjana espacial de cada canal.
#     Redueix paràmetres, regularitza implícitament i és estàndard en CNNs modernes.
#
#  4. ReduceLROnPlateau patience=2: amb epochs de ~25s, patience=5 eren >2 minuts
#     sense actualitzar el LR. Amb patience=2 reacciona molt més ràpid.

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR    = os.path.join(PROJECT_DIR, "dades", "dades")

# ================================================================
# CONFIG — modifica aquí per experimentar
# ================================================================
CONFIG = {
    # --- Dades ---
    "data_dir":   DATA_DIR,
    "batch_size": 512,

    # --- Augmentació ---
    # S'aplica només al train. Els valors estan pensats per a imatges 32×32:
    # rotació ±15°, flip horitzontal, i traslació/escala petita.
    "aug_hflip":       True,
    "aug_rotation":    15,      # graus màxims de rotació aleatòria
    "aug_translate":   0.1,     # fracció de la imatge per a traslació aleatòria
    "aug_scale":       (0.9, 1.1),  # rang d'escala aleatòria

    # --- Arquitectura ---
    # Cada element (c_mid, c_out) defineix un bloc VGG-style:
    #   Conv(c_in→c_mid, 3×3, pad=1) + BN + ReLU
    #   Conv(c_mid→c_out, 3×3, pad=1) + BN + ReLU
    #   MaxPool(2×2)                              ← substitueix la conv stride=2
    #   Dropout2d(dropout_conv)
    # Entrada 32×32: cada bloc divideix la resolució per 2.
    "conv_blocks":  [(32, 64), (128, 128), (256, 256)],  # → 16×16 → 8×8 → 4×4
    "dropout_conv": 0.2,

    # Cap de classificació: GAP → FC → ... → n_classes
    # GAP redueix cada feature map a un sol valor (la seva mitjana espacial),
    # de manera que flat_size = c_out de l'últim bloc, independent de la resolució.
    "fc_sizes":   [512, 128],
    "dropout_fc": 0.3,   # lleugerament més alt que abans per compensar el GAP

    # --- Entrenament ---
    "lr":           1e-3,
    "weight_decay": 0.0001,
    "reducelr_factor":   0.5,
    "reducelr_patience": 2,     # era 5; ara reacciona abans quan val_loss s'estanca
    "label_smooth": 0.05,
    "max_epochs":   40,
    "time_limit":   5 * 60 - 15,

    # --- Misc ---
    "seed": 0,
}
# ================================================================


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(cfg, device):
    # Transformacions de train: normalització + augmentació
    aug_list = []
    if cfg["aug_hflip"]:
        aug_list.append(transforms.RandomHorizontalFlip())
    if cfg["aug_rotation"] > 0 or cfg["aug_translate"] > 0:
        aug_list.append(transforms.RandomAffine(
            degrees=cfg["aug_rotation"],
            translate=(cfg["aug_translate"], cfg["aug_translate"]),
            scale=cfg["aug_scale"],
        ))

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        *aug_list,   # augmentació només en train
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Validació: sense augmentació, només normalització
    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.ImageFolder(os.path.join(cfg["data_dir"], "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(cfg["data_dir"], "val"),   transform=val_tf)

    pin = (device.type == "cuda")
    nw  = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


def build_model(cfg, n_classes):
    conv_layers = []
    c_in = 1
    for c_mid, c_out in cfg["conv_blocks"]:
        conv_layers += [
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            # MaxPool en lloc de Conv stride=2: més ràpid, millor invariància
            # a traslacions petites, menys paràmetres.
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(cfg["dropout_conv"]),
        ]
        c_in = c_out

    features = nn.Sequential(*conv_layers)

    # Global Average Pooling: comprimeix (B, C, H, W) → (B, C)
    # Equivalent a fer la mitjana de cada feature map sencer.
    # Avantatge: independència de resolució i fort efecte regularitzador.
    gap = nn.AdaptiveAvgPool2d(1)

    # El cap rep c_out canals (= c_in aquí, l'últim bloc)
    head_layers = [nn.Flatten()]
    fc_in = c_in   # gràcies al GAP, sempre és el nombre de canals de l'últim bloc
    for fc_out in cfg["fc_sizes"]:
        head_layers += [
            nn.Linear(fc_in, fc_out),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout_fc"]),
        ]
        fc_in = fc_out
    head_layers.append(nn.Linear(fc_in, n_classes))
    head = nn.Sequential(*head_layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.gap      = gap
            self.head     = head

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)        # (B, C, 1, 1)
            return self.head(x)    # Flatten dins head fa (B, C)

    return Net()


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


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device}\n")

    train_loader, val_loader, classes = make_loaders(CONFIG, device)
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")
    print(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}\n")

    model = build_model(CONFIG, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {n_params:,}")
    print(model)
    print()

    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smooth"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"],
                           weight_decay=CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=CONFIG["reducelr_factor"],
        patience=CONFIG["reducelr_patience"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'t(s)':>6}  {'LR':>8}")
    print("-" * 70)

    best_val_acc = 0.0
    t_start = time.time()
    stop    = False

    for epoch in range(1, CONFIG["max_epochs"] + 1):
        if stop or time.time() - t_start > CONFIG["time_limit"]:
            break

        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        pbar = tqdm(train_loader, desc=f"Època {epoch:>2}", leave=False,
                    unit="batch", dynamic_ncols=True)
        for imgs, labels in pbar:
            if time.time() - t_start > CONFIG["time_limit"]:
                stop = True
                break
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        if total == 0:
            break

        train_acc, train_loss = correct / total, loss_sum / total
        val_acc, val_loss     = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - t_start

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.0f}s  lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMillor val accuracy: {best_val_acc:.2%}")
    print(f"Temps total: {time.time() - t_start:.0f}s")