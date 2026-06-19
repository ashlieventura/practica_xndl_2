# cnn_iter.py — CNN modular per iterar arquitectura i hiperparàmetres
#
# Arquitectura base: VGG-style amb blocs de doble conv 3×3 + BN + MaxPool.
# Tot es controla des del dict CONFIG. Per experimentar, modifica:
#   - conv_blocks: llista de (c_mid, c_out) — un element = un bloc complet
#   - use_bn, dropout_conv, dropout_fc: regularització
#   - head, fc_sizes: cap de classificació
#   - optimizer, scheduler, lr, weight_decay: entrenament
#   - augment: nivell d'augmentació de dades

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image as PILImage


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "dades")

# ================================================================
# CONFIG — modifica aquí per experimentar
# ================================================================
CONFIG = {
    # --- Dades ---
    "data_dir":   DATA_DIR,
    "batch_size": 512,

    # --- Arquitectura ---
    # Cada element (c_mid, c_out) defineix un bloc complet:
    #   Conv(c_in→c_mid, 3×3, pad=1) + BN + ReLU
    #   Conv(c_mid→c_out, 3×3, pad=1) + BN + ReLU
    #   MaxPool2d(2) + Dropout2d(dropout_conv)
    # Entrada 32×32: cada bloc divideix la resolució per 2.
    "conv_blocks":  [(32, 64), (128, 256), (256, 512)],  # → 16×16 → 8×8 → 4×4
    "use_bn":       True,
    "dropout_conv": 0.15,

    # Cap de classificació
    # 'flatten': Flatten → FC → ... → n_classes
    # 'gap':     GlobalAvgPool → FC → n_classes  (menys paràmetres, prova-ho)
    "head":       "gap",
    "fc_sizes":   [256],   # capes FC intermèdies; → n_classes s'afegeix sol
    "dropout_fc": 0.3,

    # --- Entrenament ---
    "lr":           2e-3,
    "optimizer":    "adam",      # 'adam', 'adamw', 'sgd'
    "weight_decay": 0,
    "scheduler":    "onecycle",  # 'none', 'reducelr', 'onecycle', 'step'
    "reducelr_factor":   0.5,    # ReduceLROnPlateau: divideix lr per aquest factor
    "reducelr_patience": 5,      # ReduceLROnPlateau: èpoques sense millora per activar
    "label_smooth": 0.1,
    "max_epochs":  28,
    "time_limit":   15 * 60,  # temps màxim d'entrenament (s)

    # --- Augmentació ---
    # 'none' | 'light' | 'medium' | 'full'
    "augment": "light",

    # --- Misc ---
    "seed": 0,
    "cache_in_ram": True,  # True: pre-carrega tot a RAM (~144MB), elimina I/O de disc
}
# ================================================================


class InMemoryDataset(torch.utils.data.Dataset):
    """Pre-carrega totes les imatges a RAM.
    Primera execució: llegeix del disc en paral·lel (threads) i guarda caché .npy.
    Execucions posteriors: carrega el .npy directament (~1-2s fins i tot en HDD)."""
    def __init__(self, folder, transform=None):
        from concurrent.futures import ThreadPoolExecutor
        base = datasets.ImageFolder(folder)
        self.transform = transform
        self.targets   = [s[1] for s in base.samples]
        self.classes   = base.classes
        n = len(base.samples)

        cache = folder.rstrip("/\\") + "_imgs.npy"
        if os.path.exists(cache):
            print(f"  Caché trobada, carregant...", end=" ", flush=True)
            t0 = time.time()
            self._imgs = np.load(cache)
            print(f"fet en {time.time()-t0:.1f}s  ({self._imgs.nbytes//1024//1024}MB)")
        else:
            print(f"  Primera càrrega: {n:,} imatges (es crearà caché .npy per a futures runs)")
            t0 = time.time()
            paths = [p for p, _ in base.samples]

            def _read(p):
                return np.array(PILImage.open(p).convert("L"))

            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(tqdm(ex.map(_read, paths), total=n,
                                 unit="img", desc="  Llegint"))
            self._imgs = np.stack(imgs);  del imgs
            np.save(cache, self._imgs)
            print(f"  Caché guardada → {cache}  ({self._imgs.nbytes//1024//1024}MB, {time.time()-t0:.1f}s)")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = PILImage.fromarray(self._imgs[idx], mode="L")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms(augment):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if augment == "none":
        train_tf = transforms.Compose(base)
    elif augment == "light":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif augment == "medium":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif augment == "full":
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
    else:
        raise ValueError(f"augment desconegut: {augment!r}")
    return train_tf, transforms.Compose(base)


def make_loaders(cfg, device):
    train_tf, eval_tf = make_transforms(cfg["augment"])
    DS = InMemoryDataset if cfg.get("cache_in_ram", False) else datasets.ImageFolder
    train_ds = DS(os.path.join(cfg["data_dir"], "train"), transform=train_tf)
    val_ds   = DS(os.path.join(cfg["data_dir"], "val"),   transform=eval_tf)
    pin = (device.type == "cuda")
    nw = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


def build_model(cfg, n_classes):
    # Cada bloc: Conv(c_in→c_mid) + BN + ReLU + Conv(c_mid→c_out) + BN + ReLU
    #            + MaxPool2d(2) + Dropout2d
    conv_layers = []
    c_in = 1
    bias = not cfg["use_bn"]  # bias innecessari quan BN ja aprèn el seu offset
    for c_mid, c_out in cfg["conv_blocks"]:
        conv_layers.append(nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, bias=bias))
        if cfg["use_bn"]:
            conv_layers.append(nn.BatchNorm2d(c_mid))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, bias=bias))
        if cfg["use_bn"]:
            conv_layers.append(nn.BatchNorm2d(c_out))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.MaxPool2d(2, 2))
        if cfg["dropout_conv"] > 0:
            conv_layers.append(nn.Dropout2d(cfg["dropout_conv"]))
        c_in = c_out

    features = nn.Sequential(*conv_layers)

    # Càlcul automàtic de la mida del flatten
    with torch.no_grad():
        flat_size = features(torch.zeros(1, 1, 32, 32)).numel()

    head_layers = []
    if cfg["head"] == "gap":
        head_layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        fc_in = c_in
    else:
        head_layers.append(nn.Flatten())
        fc_in = flat_size

    for fc_out in cfg["fc_sizes"]:
        head_layers.append(nn.Linear(fc_in, fc_out))
        head_layers.append(nn.ReLU(inplace=True))
        if cfg["dropout_fc"] > 0:
            head_layers.append(nn.Dropout(cfg["dropout_fc"]))
        fc_in = fc_out

    head_layers.append(nn.Linear(fc_in, n_classes))
    head = nn.Sequential(*head_layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.head = head
        def forward(self, x):
            return self.head(self.features(x))

    return Net()


def build_optimizer(cfg, model):
    lr, wd = cfg["lr"], cfg["weight_decay"]
    # Separem els paràmetres: weight decay només a matrius de pesos (dim >= 2).
    # BN (gamma/beta, dim=1) i biases (dim=1) queden exempts — regularitzar-los
    # perjudica l'entrenament.
    decay     = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay  = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": wd},
              {"params": no_decay, "weight_decay": 0.0}]
    if cfg["optimizer"] == "adam":
        return optim.Adam(groups, lr=lr)
    if cfg["optimizer"] == "adamw":
        return optim.AdamW(groups, lr=lr)
    if cfg["optimizer"] == "sgd":
        return optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer desconegut: {cfg['optimizer']!r}")


def build_scheduler(cfg, optimizer, steps_per_epoch):
    if cfg["scheduler"] == "none":
        return None
    if cfg["scheduler"] == "reducelr":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min",
            factor=cfg["reducelr_factor"],
            patience=cfg["reducelr_patience"],
        )
    if cfg["scheduler"] == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"],
            steps_per_epoch=steps_per_epoch, epochs=cfg["max_epochs"],
            pct_start=0.15, div_factor=10, final_div_factor=100,
        )
    if cfg["scheduler"] == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    raise ValueError(f"scheduler desconegut: {cfg['scheduler']!r}")


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
    optimizer = build_optimizer(CONFIG, model)
    scheduler = build_scheduler(CONFIG, optimizer, len(train_loader))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

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

            if scheduler is not None and CONFIG["scheduler"] == "onecycle":
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()

            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        if total == 0:
            break

        train_acc, train_loss = correct / total, loss_sum / total
        val_acc, val_loss     = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - t_start

        # Scheduler per època: reducelr necessita val_loss, step no necessita res
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
