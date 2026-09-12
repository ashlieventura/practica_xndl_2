# cnn_v4.py — CNN sense regularització, més profunda, per explorar capacitat màxima
#
# Canvis respecte a cnn_v3:
#
#  1. SENSE regularització: dropout_conv=0, dropout_fc=0, weight_decay=0,
#     label_smooth=0. L'objectiu és primer veure si la xarxa pot aprendre
#     (overfit fort = bona senyal), i després afegir regularització al pas següent.
#
#  2. Convs extra per bloc (extra_convs): afegim convolutions addicionals SENSE
#     downsampling (padding=1 manté la resolució) dins de cada bloc, abans del
#     downsampling final. Això dóna a la xarxa més capacitat per aprendre
#     representacions riques a cada escala espacial.
#     extra_convs=0 → 2 convs/bloc (original)
#     extra_convs=1 → 3 convs/bloc (VGG-style)
#     extra_convs=2 → 4 convs/bloc (VGG16-style)
#
#  3. Downsampling híbrid heretat de v3: MaxPool pels primers blocs,
#     Conv stride=2 pels últims.

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
    "aug_hflip":     True,
    "aug_rotation":  15,
    "aug_translate": 0.1,
    "aug_scale":     (0.9, 1.1),

    # --- Arquitectura ---
    # Cada element (c_mid, c_out) defineix un bloc:
    #   Conv(c_in→c_mid, 3×3, pad=1) + BN + ReLU
    #   [Conv(c_mid→c_mid, 3×3, pad=1) + BN + ReLU] × extra_convs  ← SENSE downsampling
    #   Conv(c_mid→c_out, 3×3, pad=1) + BN + ReLU
    #   Downsampling (MaxPool o Conv stride=2)
    #
    # Les convs extra usen padding=1 → la resolució espacial NO canvia.
    # Permeten aprendre features més complexes a cada escala.
    #   extra_convs=0 → 2 convs per bloc
    #   extra_convs=1 → 3 convs per bloc (VGG-style, recomanat)
    #   extra_convs=2 → 4 convs per bloc (més capacitat, més lent)
    "conv_blocks":    [(32, 64), (128, 128), (256, 256)],
    "extra_convs":    1,    # convs extra per bloc sense downsampling
    "maxpool_blocks": 2,    # primers N blocs amb MaxPool, la resta Conv stride=2
    "dropout_conv":   0.0,  # sense regularització per veure capacitat màxima

    # Cap de classificació: GAP → FC → ... → n_classes
    "fc_sizes":    [512, 256],
    "dropout_fc":  0.0,

    # --- Entrenament --- sense regularització
    "lr":                1e-3,
    "weight_decay":      0.0,
    "reducelr_factor":   0.5,
    "reducelr_patience": 1,
    "label_smooth":      0.0,
    "max_epochs":        40,
    "time_limit":        10 * 60,

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
        *aug_list,
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
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
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


def build_model(cfg, n_classes):
    conv_layers = []
    c_in      = 1
    n_maxpool = cfg.get("maxpool_blocks", len(cfg["conv_blocks"]))
    n_extra   = cfg.get("extra_convs", 0)

    for i, (c_mid, c_out) in enumerate(cfg["conv_blocks"]):
        # Downsampling: MaxPool pels primers blocs (invariança translacional),
        # Conv stride=2 pels últims (la xarxa aprèn com comprimir features abstractes).
        if i < n_maxpool:
            downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            downsample = nn.Sequential(
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, stride=2),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        # Primera conv: puja els canals de c_in a c_mid
        bloc = [
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
        ]

        # Convs extra: mantenen c_mid canals i NO fan downsampling (padding=1).
        # Cada conv extra permet aprendre combinacions més complexes dels features
        # actuals sense perdre resolució espacial.
        for _ in range(n_extra):
            bloc += [
                nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_mid),
                nn.ReLU(inplace=True),
            ]

        # Última conv del bloc: porta de c_mid a c_out i fa el downsampling
        bloc += [
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            downsample,
        ]

        # Dropout2d només si s'ha configurat (ara és 0.0, però deixem el mecanisme)
        if cfg.get("dropout_conv", 0.0) > 0:
            bloc.append(nn.Dropout2d(cfg["dropout_conv"]))

        conv_layers += bloc
        c_in = c_out

    features = nn.Sequential(*conv_layers)

    # Global Average Pooling: (B, C, H, W) → (B, C, 1, 1)
    # Independència de resolució i regularització implícita.
    gap = nn.AdaptiveAvgPool2d(1)

    head_layers = [nn.Flatten()]
    fc_in = c_in
    for fc_out in cfg["fc_sizes"]:
        head_layers += [nn.Linear(fc_in, fc_out), nn.ReLU(inplace=True)]
        if cfg.get("dropout_fc", 0.0) > 0:
            head_layers.append(nn.Dropout(cfg["dropout_fc"]))
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
            x = self.gap(x)
            return self.head(x)

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