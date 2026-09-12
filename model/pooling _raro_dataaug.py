# model_final.py — CNN per a classificació d'imatges 32×32 en 14 categories
#
# Arquitectura: VGG-style, 3 blocs conv doble + BN, conv strided al bloc final.
# Augmentació: seleccionable via AUGMENT (vegeu més avall).

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor

# ================================================================
# CONFIGURACIÓ — modifica aquí
# ================================================================

DATA_DIR    = "../dades"
BATCH_SIZE  = 512
LR          = 2e-3
MAX_EPOCHS  = 28
TIME_LIMIT  = 5 * 60   # segons; restricció de l'avaluació
LABEL_SMOOTH = 0.1
SEED        = 0

# Arquitectura
CONV_BLOCKS  = [(32, 64), (128, 256), (256, 512)]  # (c_mid, c_out) per bloc
DROPOUT_CONV = 0.1   # Dropout2d després de cada bloc
DROPOUT_FC   = 0.3   # Dropout a la capa FC

# Augmentació — tria una de les 4 opcions:
#   "none"    → sense augmentació (baseline)
#   "light"   → rotació + translació lleugera
#   "erasing" → light + RandomErasing (zones aleatòries esborrades)
#   "cutout"  → light + CutOut centrat (tapa el centre on sol estar l'interior)
AUGMENT = "light"

# Paràmetres d'augmentació — ajusta després del grid search
ROTATION      = 10              # graus màxims de rotació (light, erasing, cutout)
TRANSLATE     = (0.06, 0.06)   # translació màxima com a fracció (light, erasing, cutout)
ERASING_SCALE = (0.02, 0.10)   # fracció d'àrea esborrada per RandomErasing
ERASING_RATIO = (0.3, 3.0)     # ràtio d'aspecte del parxo d'erasing
ERASING_P     = 0.5            # probabilitat d'aplicar RandomErasing
CUTOUT_SIZE   = 10             # costat del parxo en píxels (sobre 32×32)
CUTOUT_P      = 0.5            # probabilitat d'aplicar CutOut

# ================================================================


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ================================================================
# AUGMENTACIÓ
# ================================================================

class CutOutTransform:
    """Tapa una finestra quadrada centrada amb valor 0 (mitjà després de normalitzar).
    A diferència de RandomErasing, el parxo és sempre al centre — on sol estar
    l'interior dels objectes rodons — forçant el model a usar la perifèria."""
    def __init__(self, size, p=0.5):
        self.size = size
        self.p    = p

    def __call__(self, img):
        # img és un tensor [C, H, W] ja normalitzat
        if random.random() > self.p:
            return img
        _, h, w = img.shape
        cy, cx  = h // 2, w // 2
        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


def make_transforms():
    """Retorna (train_transform, eval_transform) segons AUGMENT."""

    # Pipeline base (sempre igual per a eval i com a punt de partida per a train)
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]

    if AUGMENT == "none":
        # Sense cap augmentació: el model veu exactament les imatges originals.
        # Útil com a baseline per mesurar quant aporta cadascuna de les altres.
        train_tf = transforms.Compose(base)

    elif AUGMENT == "light":
        # Rotació aleatòria ± ROTATION graus + translació fins a TRANSLATE.
        # Fa el model invariant a petits desplaçaments i orientacions,
        # sense distorsionar el contingut intern.
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(ROTATION),
            transforms.RandomAffine(degrees=0, translate=TRANSLATE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    elif AUGMENT == "erasing":
        # Light + RandomErasing: borra una zona rectangular aleatòria.
        # Obliga el model a no dependre d'un sol lloc de l'interior per
        # classificar — clau quan les classes es distingeixen pel que hi ha dins.
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(ROTATION),
            transforms.RandomAffine(degrees=0, translate=TRANSLATE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(
                p=ERASING_P,
                scale=ERASING_SCALE,
                ratio=ERASING_RATIO,
                value=0,   # 0 = valor mitjà després de normalitzar a [-1,1]
            ),
        ])

    elif AUGMENT == "cutout":
        # Light + CutOut centrat: tapa sempre el centre de la imatge.
        # Més agressiu que erasing perquè ataca directament la zona
        # més discriminativa (l'interior de l'objecte rodó).
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(ROTATION),
            transforms.RandomAffine(degrees=0, translate=TRANSLATE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            CutOutTransform(size=CUTOUT_SIZE, p=CUTOUT_P),
        ])

    else:
        raise ValueError(f"AUGMENT desconegut: {AUGMENT!r}. Tria: none, light, erasing, cutout")

    eval_tf = transforms.Compose(base)
    return train_tf, eval_tf


# ================================================================
# DADES AMB CACHÉ EN RAM
# ================================================================

class InMemoryDataset(torch.utils.data.Dataset):
    """Pre-carrega totes les imatges a RAM.
    Primera execució: llegeix del disc i crea un .npy de caché.
    Execucions posteriors: carrega el .npy directament (~1-2s)."""
    def __init__(self, folder, transform=None):
        base           = datasets.ImageFolder(folder)
        self.transform = transform
        self.targets   = [s[1] for s in base.samples]
        self.classes   = base.classes
        cache          = folder.rstrip("/\\") + "_imgs.npy"

        if os.path.exists(cache):
            print(f"  Caché trobada: {cache}", flush=True)
            self._imgs = np.load(cache)
        else:
            print(f"  Primera càrrega de {folder} (es crearà caché .npy)...")
            paths = [p for p, _ in base.samples]
            def _read(p): return np.array(PILImage.open(p).convert("L"))
            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(tqdm(ex.map(_read, paths), total=len(paths), unit="img"))
            self._imgs = np.stack(imgs)
            np.save(cache, self._imgs)

    def __len__(self): return len(self.targets)

    def __getitem__(self, idx):
        img = PILImage.fromarray(self._imgs[idx], mode="L")
        if self.transform:
            img = self.transform(img)
        return img, self.targets[idx]


def make_loaders():
    train_tf, eval_tf = make_transforms()
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_DIR)
    train_ds = InMemoryDataset(os.path.join(base_dir, "train"), transform=train_tf)
    val_ds   = InMemoryDataset(os.path.join(base_dir, "val"),   transform=eval_tf)
    pin = (device.type == "cuda")
    nw  = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


# ================================================================
# ARQUITECTURA
# ================================================================

def build_model(n_classes):
    """CNN VGG-style: N blocs de doble conv 3×3 + BN + ReLU + downsampling.
    Tots els blocs menys l'últim fan MaxPool2d(2).
    L'últim bloc fa una conv strided (stride=2) en lloc de MaxPool:
    el downsampling és après, no fix, útil quan la resolució ja és petita (8→4)."""
    n_blocks = len(CONV_BLOCKS)
    bias     = False  # BN té el seu propi offset; bias redundant
    c_in     = 1
    layers   = []

    for i, (c_mid, c_out) in enumerate(CONV_BLOCKS):
        # Primera conv del bloc
        layers += [nn.Conv2d(c_in, c_mid, 3, padding=1, bias=bias),
                   nn.BatchNorm2d(c_mid),
                   nn.ReLU(inplace=True)]
        # Segona conv del bloc
        layers += [nn.Conv2d(c_mid, c_out, 3, padding=1, bias=bias),
                   nn.BatchNorm2d(c_out),
                   nn.ReLU(inplace=True)]
        # Downsampling
        if i < n_blocks - 1:
            layers.append(nn.MaxPool2d(2, 2))
        else:
            layers += [nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, bias=bias),
                       nn.BatchNorm2d(c_out),
                       nn.ReLU(inplace=True)]
        # Regularització espacial
        if DROPOUT_CONV > 0:
            layers.append(nn.Dropout2d(DROPOUT_CONV))
        c_in = c_out

    features = nn.Sequential(*layers)
    c_last   = CONV_BLOCKS[-1][1]

    # Cap: Global Average Pool → FC → Dropout → FC de sortida
    # GAP redueix cada mapa de característiques a un escalar,
    # eliminant paràmetres i afegint invariància a la posició.
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(c_last, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(DROPOUT_FC),
        nn.Linear(256, n_classes),
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.head     = head
        def forward(self, x):
            return self.head(self.features(x))

    return Net()


# ================================================================
# ENTRENAMENT
# ================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, use_amp):
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
    print(f"Augmentació: {AUGMENT}\n")

    print("Carregant dades...")
    train_loader, val_loader, classes = make_loaders()
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")
    print(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}\n")

    model    = build_model(n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {n_params:,}\n")

    use_amp   = (device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    # Separem weight decay: només a matrius de pesos (dim≥2), mai a BN ni biases
    decay    = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = optim.Adam([{"params": decay,    "weight_decay": 0},
                             {"params": no_decay, "weight_decay": 0}], lr=LR)

    # OneCycleLR: puja el lr fins a LR durant el 15% del temps,
    # després baixa suaument. Convergeix ràpid i sense necessitat de tuning manual.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=len(train_loader), epochs=MAX_EPOCHS,
        pct_start=0.15, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"{'Epoch':>5}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'t(s)':>6}")
    print("─" * 60)

    best_val_acc = 0.0
    t_start      = time.time()
    stop         = False

    for epoch in range(1, MAX_EPOCHS + 1):
        if stop or time.time() - t_start > TIME_LIMIT:
            print(f"Temps exhaurit abans de l'epoch {epoch}. Aturant.")
            break

        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for imgs, labels in train_loader:
            if time.time() - t_start > TIME_LIMIT:
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
            if scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()
            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)

        if total == 0: break
        train_acc, train_loss = correct / total, loss_sum / total
        val_acc, val_loss     = evaluate(model, val_loader, criterion, use_amp)
        elapsed               = time.time() - t_start

        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.0f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"\nMillor val accuracy: {best_val_acc:.2%}")
    print(f"Temps total: {time.time() - t_start:.0f}s")