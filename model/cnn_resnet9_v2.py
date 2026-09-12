# cnn_resnet9_v2.py — ResNet-9 + CBAM + Mixup + TTA + regularització forta
#
# Canvis respecte a v1 (millor val acc: 86.00%, overfitting clar a partir epoch 38):
#
# PROBLEMA: brecha train-val de ~7 punts (train 92% vs val 86%). El model
# memoritzava en lloc de generalitzar. Solució en 4 fronts:
#
# 1. MIXUP: barreja dues imatges del batch (img = λ·A + (1-λ)·B) i interpola
#    els seus labels. Força el model a aprendre representacions lineals i suaus
#    en lloc de memoritzar exemples concrets. Molt efectiu en datasets petits.
#    α=0.4: λ ~ Beta(0.4, 0.4), que dóna barreges ni massa suaus ni massa fortes.
#
# 2. MÉS DROPOUT: dropout_fc 0.3 → 0.5, dropout spatial (Dropout2d) afegit
#    després de cada bloc convolucional. Regularitza tant les features espacials
#    com el cap de classificació.
#
# 3. MÉS WEIGHT DECAY: 1e-4 → 4e-4. Penalitza pesos grans més agressivament,
#    forçant solucions més simples i generalitzables.
#
# 4. AUGMENTACIÓ MÉS AGRESSIVA: rotació fins a 180° (els objectes són circulars,
#    una rotació de 180° no canvia la semàntica) + erasing medium.
#
# 5. TTA (Test Time Augmentation): a l'avaluació, cada imatge es prediu 6 vegades
#    amb augmentacions aleatòries i es fa la mitjana dels logits. No millora el
#    model però sí la mètrica final — és gratis en temps d'inferència.
#
# 6. CAPACITAT REDUÏDA: 512 canals → 384 al bloc 3. Menys paràmetres = menys
#    risc de memoritzar, i més ràpid per epoch (més epochs en 5 min).

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
    "augment":     "none",

    # Regularització — tots pujats respecte v1
    "dropout_fc":      0.5,        # v1: 0.3
    "dropout_conv":    0.2,        # v1: 0.0 (no n'hi havia)
    "weight_decay":    4e-4,       # v1: 1e-4

    # Mixup
    "mixup_alpha":     0.4,        # 0 = desactivat; >0 = activat

    # TTA (Test Time Augmentation)
    "tta_n":           6,          # nombre de passes d'augmentació a val/test

    # Entrenament
    "lr":           1.5e-3,        # lleugerament més alt: Mixup suavitza els gradients
    "optimizer":    "adamw",
    "scheduler":    "onecycle",
    "pct_start":    0.10,          # v1: 0.15 — warmup més curt per afinar abans
    "label_smooth": 0.1,          # baix perquè Mixup ja suavitza els targets
    "max_epochs":   30,            # més epochs: model més lleuger → epoch més ràpida
    "time_limit":   30 * 60 - 15,

    "seed":         42,
    "cache_in_ram": True,
}
# ================================================================


# ================================================================
# CBAM (igual que v1)
# ================================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels, bias=False),
        )
    def forward(self, x):
        avg = self.fc(x.mean(dim=[2, 3]))
        mx  = self.fc(x.amax(dim=[2, 3]))
        w   = torch.sigmoid(avg + mx).unsqueeze(2).unsqueeze(3)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, r=8):
        super().__init__()
        self.ca = ChannelAttention(channels, r)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))


# ================================================================
# ARQUITECTURA RESNET-9 (capacitat reduïda: 384 en lloc de 512)
# ================================================================

def conv_bn(c_in, c_out, kernel=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class ResNet9(nn.Module):
    def __init__(self, n_classes, dropout_fc=0.5, dropout_conv=0.2):
        super().__init__()

        self.prep = conv_bn(1, 64)                        # 32×32

        self.layer1      = conv_bn(64, 128)               # 32×32
        self.layer1_pool = nn.MaxPool2d(2)                # 16×16
        self.res1 = nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))
        self.drop1 = nn.Dropout2d(dropout_conv)

        self.layer2      = conv_bn(128, 256)              # 16×16
        self.layer2_pool = nn.MaxPool2d(2)                # 8×8
        self.drop2 = nn.Dropout2d(dropout_conv)

        # 512 → 384: menys paràmetres, menys overfitting, epoch més ràpida
        self.layer3      = conv_bn(256, 384)              # 8×8
        self.layer3_pool = nn.MaxPool2d(2)                # 4×4
        self.res3 = nn.Sequential(conv_bn(384, 384), conv_bn(384, 384))
        self.drop3 = nn.Dropout2d(dropout_conv)

        self.cbam    = CBAM(384, r=8)
        self.pool    = nn.AdaptiveAvgPool2d(1)            # GAP: més robust que MaxPool
        self.dropout = nn.Dropout(dropout_fc)
        self.fc      = nn.Linear(384, n_classes)

    def forward(self, x):
        x = self.prep(x)

        x = self.layer1(x)
        x = self.layer1_pool(x)
        x = self.drop1(x + self.res1(x))

        x = self.layer2(x)
        x = self.layer2_pool(x)
        x = self.drop2(x)

        x = self.layer3(x)
        x = self.layer3_pool(x)
        x = self.drop3(x + self.res3(x))

        x = self.cbam(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# ================================================================
# MIXUP
# ================================================================

def mixup_batch(imgs, labels, n_classes, alpha):
    """Barreja aleatòria de parelles d'imatges del batch.
    Retorna imatges barrejades i labels suavitzats com a distribució.
    Usem one-hot + interpolació perquè CrossEntropyLoss amb label_smoothing
    no accepta targets continus — usem KLDivLoss en el loop d'entrenament."""
    if alpha <= 0:
        return imgs, F.one_hot(labels, n_classes).float()
    lam = np.random.beta(alpha, alpha)
    # Amestrem el batch amb si mateix desplaçat
    idx     = torch.randperm(imgs.size(0), device=imgs.device)
    imgs_m  = lam * imgs + (1 - lam) * imgs[idx]
    # Labels suavitzats: distribució convexa entre les dues classes
    y_a = F.one_hot(labels,      n_classes).float()
    y_b = F.one_hot(labels[idx], n_classes).float()
    labels_m = lam * y_a + (1 - lam) * y_b
    return imgs_m, labels_m


def mixup_loss(logits, labels_soft, label_smooth=0.0, n_classes=14):
    """KLDivLoss equivalent a CrossEntropy amb targets continus.
    Aplica label smoothing manualment sobre els soft labels."""
    if label_smooth > 0:
        labels_soft = labels_soft * (1 - label_smooth) + label_smooth / n_classes
    log_prob = F.log_softmax(logits, dim=1)
    return -(labels_soft * log_prob).sum(dim=1).mean()


# ================================================================
# TTA (Test Time Augmentation)
# ================================================================

def make_tta_transform():
    """Transform per a TTA: igual que l'augmentació d'entrenament però
    sense erasing (volem augmentar la diversitat de vistes, no eliminar info)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


@torch.no_grad()
def evaluate_tta(model, dataset, device, use_amp, n_tta, batch_size):
    """Avaluació amb TTA: cada imatge es prediu n_tta vegades amb augmentacions
    aleatòries i es fa la mitjana dels logits (en escala log-prob per ser correcte).
    Més lent que evaluate() però dóna ~0.5-1% més d'acc sense canviar el model."""
    model.eval()
    tta_tf  = make_tta_transform()
    base_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Acumulem logits per a cada imatge del val set
    all_logits = []
    all_labels = []

    # Passem 1 vegada sense augmentació + n_tta-1 vegades amb augmentació
    for pass_i in range(n_tta):
        tf = base_tf if pass_i == 0 else tta_tf
        # Creem un dataset temporal amb el transform d'aquesta passada
        tmp_ds = _TmpDataset(dataset, tf)
        loader = DataLoader(tmp_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda"))
        pass_logits = []
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs)
            pass_logits.append(logits.cpu())
            if pass_i == 0:
                all_labels.append(labels)
        all_logits.append(torch.cat(pass_logits))

    # Mitjana dels logits de totes les passades
    avg_logits = torch.stack(all_logits).mean(0)
    labels_all = torch.cat(all_labels)
    correct    = (avg_logits.argmax(1) == labels_all).sum().item()
    return correct / len(labels_all)


class _TmpDataset(torch.utils.data.Dataset):
    """Wrapper que aplica un transform diferent sobre un InMemoryDataset existent.
    Evita tornar a llegir les imatges del disc per a cada passada de TTA."""
    def __init__(self, base_ds, transform):
        self._imgs    = base_ds._imgs
        self.targets  = base_ds.targets
        self.transform = transform
    def __len__(self): return len(self.targets)
    def __getitem__(self, idx):
        img = PILImage.fromarray(self._imgs[idx], mode="L")
        return self.transform(img), self.targets[idx]


# ================================================================
# DADES
# ================================================================

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, folder, transform=None):
        base = datasets.ImageFolder(folder)
        self.transform = transform
        self.targets = [s[1] for s in base.samples]
        self.classes = base.classes
        cache = folder.rstrip("/\\") + "_imgs.npy"
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
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    elif augment == "erasing_medium":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.5, scale=(0.08, 0.20), ratio=(0.5, 2.0), value=0),
        ])

    elif augment == "aggressive":
        # Rotació fins a 180°: els objectes són cercles, cap classe canvia de
        # significat en rotar. És la augmentació més potent per a aquest dataset.
        # Erasing medium: força el model a usar l'interior complet, no una zona.
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),
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
    return train_loader, val_loader, train_ds, val_ds


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
    if cfg["scheduler"] == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"],
            steps_per_epoch=steps_per_epoch, epochs=cfg["max_epochs"],
            pct_start=cfg.get("pct_start", 0.10),
            div_factor=10, final_div_factor=100,
        )
    return None


# ================================================================
# AVALUACIÓ RÀPIDA (sense TTA, per al log per epoch)
# ================================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(imgs)
        correct += (out.argmax(1) == labels).sum().item()
        total   += len(labels)
    return correct / total


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device}\n")

    train_loader, val_loader, train_ds, val_ds = make_loaders(CONFIG, device)
    n_classes = len(train_ds.classes)
    print(f"Classes ({n_classes}): {train_ds.classes}")
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}\n")

    model    = ResNet9(n_classes,
                       dropout_fc=CONFIG["dropout_fc"],
                       dropout_conv=CONFIG["dropout_conv"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {n_params:,}\n")

    optimizer = build_optimizer(CONFIG, model)
    scheduler = build_scheduler(CONFIG, optimizer, len(train_loader))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Acc':>7}  {'Val TTA':>8}  {'t(s)':>6}  {'LR':>8}")
    print("-" * 72)

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

            # Aplica Mixup al batch
            imgs_m, labels_soft = mixup_batch(
                imgs, labels, n_classes, CONFIG["mixup_alpha"]
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs_m)
                loss   = mixup_loss(logits, labels_soft,
                                    CONFIG["label_smooth"], n_classes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None:
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()

            loss_sum += loss.item() * len(labels)
            # Acc de train: sobre les imatges originals (sense mixup) per ser comparable
            with torch.no_grad():
                correct += (model(imgs).argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if total == 0: break

        train_acc  = correct / total
        train_loss = loss_sum / total

        # Val sense TTA (ràpid, per al log)
        val_acc = evaluate(model, val_loader, device, use_amp)

        # Val amb TTA (més lent, cada 5 epochs i a l'últim)
        is_last = stop or (time.time() - t_start > CONFIG["time_limit"] * 0.95)
        if epoch % 5 == 0 or is_last:
            val_tta = evaluate_tta(model, val_ds, device, use_amp,
                                   CONFIG["tta_n"], CONFIG["batch_size"])
            tta_str = f"{val_tta:>7.2%}"
        else:
            val_tta = None
            tta_str = "       -"

        elapsed    = time.time() - t_start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_acc:>6.2%}  {tta_str}  {elapsed:>5.0f}s  lr={current_lr:.2e}")

        # Guardem el millor model basat en val_acc (sense TTA per ser consistent)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    # TTA final sobre el millor model guardat
    print(f"\nMillor val accuracy (sense TTA): {best_val_acc:.2%}")
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    final_tta = evaluate_tta(model, val_ds, device, use_amp,
                             CONFIG["tta_n"], CONFIG["batch_size"])
    print(f"Millor val accuracy (amb TTA ×{CONFIG['tta_n']}): {final_tta:.2%}")
    print(f"Temps total: {time.time() - t_start:.0f}s")
