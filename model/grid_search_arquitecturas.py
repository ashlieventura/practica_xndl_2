# grid_search.py — Cerca d'hiperparàmetres per a la CNN de classificació 32×32
#
# Ús:
#   python grid_search.py                        # ordinador únic
#   python grid_search.py --n_workers 3 --worker_id 0   # màquina 0 de 3
#   python grid_search.py --n_workers 3 --worker_id 1   # màquina 1 de 3
#   python grid_search.py --n_workers 3 --worker_id 2   # màquina 2 de 3
#
# Cada màquina entrena el seu subconjunt de configs i desa els resultats
# en un CSV propi (results_worker0.csv, etc.). Quan totes han acabat,
# ajunta-les amb:
#   python grid_search.py --merge

import os, time, csv, json, argparse, random, itertools
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
# ESPAI DE CERCA — modifica aquí per ampliar o reduir la cerca
# ================================================================
SEARCH_SPACE = {
    "conv_blocks": [
        [(32, 64),  (128, 256), (256, 512)],   # config base
        [(32, 64),  (128, 256), (256, 256)],   # bloc 3 més lleuger
        [(64, 128), (128, 256), (256, 512)],   # bloc 1 més potent
    ],
    "dropout_conv": [0.0, 0.1, 0.2],
    "dropout_fc":   [0.2, 0.4],
}

# Hiperparàmetres fixos (no es cerquen)
FIXED = {
    "data_dir":          "../dades",
    "batch_size":        512,
    "head":              "gap",
    "fc_sizes":          [256],
    "use_bn":            True,
    "lr":                2e-3,
    "optimizer":         "adam",
    "weight_decay":      0,
    "scheduler":         "onecycle",
    "reducelr_factor":   0.5,
    "reducelr_patience": 5,
    "label_smooth":      0.1,
    "max_epochs":        28,
    "time_limit":        5 * 60,   # 5 min per model, igual que l'avaluació
    "augment":           "light",
    "seed":              0,
    "cache_in_ram":      True,
}

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)


# ================================================================
# UTILITATS DE DADES
# ================================================================

class InMemoryDataset(torch.utils.data.Dataset):
    """Pre-carrega totes les imatges a RAM amb caché .npy per a runs posteriors."""
    def __init__(self, folder, transform=None):
        base = datasets.ImageFolder(folder)
        self.transform = transform
        self.targets   = [s[1] for s in base.samples]
        self.classes   = base.classes
        n = len(base.samples)
        cache = folder.rstrip("/\\") + "_imgs.npy"
        if os.path.exists(cache):
            self._imgs = np.load(cache)
        else:
            paths = [p for p, _ in base.samples]
            def _read(p): return np.array(PILImage.open(p).convert("L"))
            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(tqdm(ex.map(_read, paths), total=n, desc="Carregant dades"))
            self._imgs = np.stack(imgs)
            np.save(cache, self._imgs)

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
    if augment == "light":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        train_tf = transforms.Compose(base)
    return train_tf, transforms.Compose(base)


def make_loaders(cfg, device):
    train_tf, eval_tf = make_transforms(cfg["augment"])
    data_dir = os.path.join(PROJECT_DIR, cfg["data_dir"].lstrip("./"))
    DS = InMemoryDataset if cfg.get("cache_in_ram") else datasets.ImageFolder
    train_ds = DS(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = DS(os.path.join(data_dir, "val"),   transform=eval_tf)
    pin = (device.type == "cuda")
    nw  = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


# ================================================================
# ARQUITECTURA — igual que cnn_iter.py però amb conv strided al bloc final
# ================================================================

def build_model(cfg, n_classes):
    conv_layers = []
    c_in   = 1
    bias   = not cfg["use_bn"]
    n_blocks = len(cfg["conv_blocks"])

    for i, (c_mid, c_out) in enumerate(cfg["conv_blocks"]):
        conv_layers.append(nn.Conv2d(c_in, c_mid, 3, padding=1, bias=bias))
        if cfg["use_bn"]: conv_layers.append(nn.BatchNorm2d(c_mid))
        conv_layers.append(nn.ReLU(inplace=True))

        conv_layers.append(nn.Conv2d(c_mid, c_out, 3, padding=1, bias=bias))
        if cfg["use_bn"]: conv_layers.append(nn.BatchNorm2d(c_out))
        conv_layers.append(nn.ReLU(inplace=True))

        # Blocs 1..n-1: MaxPool estàndard.
        # Bloc final: conv strided 3×3 — el downsampling és après, no fix.
        if i < n_blocks - 1:
            conv_layers.append(nn.MaxPool2d(2, 2))
        else:
            conv_layers.append(nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, bias=bias))
            if cfg["use_bn"]: conv_layers.append(nn.BatchNorm2d(c_out))
            conv_layers.append(nn.ReLU(inplace=True))

        if cfg["dropout_conv"] > 0:
            conv_layers.append(nn.Dropout2d(cfg["dropout_conv"]))
        c_in = c_out

    features = nn.Sequential(*conv_layers)

    # Cap de classificació (GAP per defecte)
    c_last = cfg["conv_blocks"][-1][1]
    head_layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    fc_in = c_last
    for fc_out in cfg["fc_sizes"]:
        head_layers += [nn.Linear(fc_in, fc_out), nn.ReLU(inplace=True)]
        if cfg["dropout_fc"] > 0:
            head_layers.append(nn.Dropout(cfg["dropout_fc"]))
        fc_in = fc_out
    head_layers.append(nn.Linear(fc_in, n_classes))
    head = nn.Sequential(*head_layers)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.head     = head
        def forward(self, x): return self.head(self.features(x))

    return Net()


def build_optimizer(cfg, model):
    lr, wd   = cfg["lr"], cfg["weight_decay"]
    decay    = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups   = [{"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0}]
    if cfg["optimizer"] == "adam":  return optim.Adam(groups, lr=lr)
    if cfg["optimizer"] == "adamw": return optim.AdamW(groups, lr=lr)
    if cfg["optimizer"] == "sgd":   return optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True)
    raise ValueError(cfg["optimizer"])


def build_scheduler(cfg, optimizer, steps_per_epoch):
    if cfg["scheduler"] == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"],
            steps_per_epoch=steps_per_epoch, epochs=cfg["max_epochs"],
            pct_start=0.15, div_factor=10, final_div_factor=100,
        )
    return None


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
# ENTRENAMENT D'UNA SOLA CONFIG
# ================================================================

def train_config(cfg, train_loader, val_loader, n_classes, device, config_id):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    use_amp  = (device.type == "cuda")
    model    = build_model(cfg, n_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smooth"])
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    t_start      = time.time()
    stop         = False

    for epoch in range(1, cfg["max_epochs"] + 1):
        if stop or time.time() - t_start > cfg["time_limit"]:
            break

        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for imgs, labels in train_loader:
            if time.time() - t_start > cfg["time_limit"]:
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
            if scheduler is not None and scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()
            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)

        if total == 0: break
        val_acc, val_loss = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - t_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_model_cfg{config_id}.pt")

        print(f"  [cfg {config_id}] epoch {epoch:>2}  "
              f"val_acc={val_acc:.2%}  best={best_val_acc:.2%}  t={elapsed:.0f}s")

    total_time = time.time() - t_start
    return best_val_acc, total_time, n_params


# ================================================================
# GRID SEARCH AMB REPARTIMENT EQUITATIU ENTRE MÀQUINES
# ================================================================

def build_grid():
    """Genera totes les combinacions de l'espai de cerca."""
    keys   = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    combos = []
    for combo in itertools.product(*values):
        cfg = {**FIXED, **dict(zip(keys, combo))}
        combos.append(cfg)
    return combos


def assign_configs(all_configs, n_workers, worker_id):
    """Reparteix les configs entre màquines de forma equitativa (round-robin).
    Round-robin garanteix que cada màquina té aproximadament el mateix nombre
    de configs independentment de si el total és divisible per n_workers."""
    return [cfg for i, cfg in enumerate(all_configs) if i % n_workers == worker_id]


def save_result(csv_path, config_id, cfg, val_acc, elapsed, n_params):
    """Afegeix una fila al CSV de resultats. Crea la capçalera si el fitxer no existeix."""
    fieldnames = ["config_id", "val_acc", "elapsed_s", "n_params",
                  "conv_blocks", "dropout_conv", "dropout_fc"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "config_id":    config_id,
            "val_acc":      f"{val_acc:.4f}",
            "elapsed_s":    f"{elapsed:.0f}",
            "n_params":     n_params,
            "conv_blocks":  json.dumps(cfg["conv_blocks"]),
            "dropout_conv": cfg["dropout_conv"],
            "dropout_fc":   cfg["dropout_fc"],
        })


def merge_results(n_workers):
    """Ajunta tots els CSV parcials i imprimeix la taula ordenada per val_acc."""
    rows = []
    for wid in range(n_workers):
        path = f"results_worker{wid}.csv"
        if not os.path.exists(path):
            print(f"  Avís: no trobat {path}")
            continue
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))

    if not rows:
        print("No hi ha resultats per ajuntar.")
        return

    rows.sort(key=lambda r: float(r["val_acc"]), reverse=True)

    print(f"\n{'Rank':>4}  {'val_acc':>8}  {'drop_conv':>9}  {'drop_fc':>7}  "
          f"{'n_params':>9}  {'t(s)':>5}  conv_blocks")
    print("─" * 85)
    for rank, r in enumerate(rows, 1):
        blocks = json.loads(r["conv_blocks"])
        print(f"{rank:>4}  {float(r['val_acc']):>7.2%}  "
              f"{float(r['dropout_conv']):>9.2f}  {float(r['dropout_fc']):>7.2f}  "
              f"{int(r['n_params']):>9,}  {int(float(r['elapsed_s'])):>5}  {blocks}")

    # Desa també el merge complet
    with open("results_merged.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultats complets desats a results_merged.csv")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search paral·lelitzable entre màquines")
    parser.add_argument("--n_workers",  type=int, default=1,
                        help="Nombre total d'ordinadors/workers (default: 1)")
    parser.add_argument("--worker_id",  type=int, default=0,
                        help="ID d'aquest ordinador, de 0 a n_workers-1 (default: 0)")
    parser.add_argument("--merge",      action="store_true",
                        help="Ajunta els CSV parcials i imprimeix la taula de resultats")
    args = parser.parse_args()

    # Mode merge: ajunta resultats de tots els workers i surt
    if args.merge:
        # Detecta automàticament quants workers hi ha pels fitxers CSV existents
        existing = [f for f in os.listdir(".") if f.startswith("results_worker") and f.endswith(".csv")]
        n = max(int(f.replace("results_worker","").replace(".csv","")) for f in existing) + 1 if existing else args.n_workers
        merge_results(n)
        exit(0)

    # Validació d'arguments
    assert 0 <= args.worker_id < args.n_workers, \
        f"worker_id ({args.worker_id}) ha de ser entre 0 i n_workers-1 ({args.n_workers-1})"

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    print(f"Device: {device}")
    print(f"Worker {args.worker_id} de {args.n_workers}\n")

    # Construeix la grid i assigna configs a aquest worker
    all_configs    = build_grid()
    my_configs     = assign_configs(all_configs, args.n_workers, args.worker_id)
    global_indices = [i for i in range(len(all_configs)) if i % args.n_workers == args.worker_id]

    print(f"Total de configs: {len(all_configs)}")
    print(f"Configs d'aquest worker: {len(my_configs)} "
          f"(IDs globals: {global_indices})\n")

    # Càrrega de dades una sola vegada per a tots els models d'aquest worker
    # (estalvia temps de I/O reutilitzant els loaders entre configs)
    print("Carregant dades...")
    train_loader, val_loader, classes = make_loaders({**FIXED}, device)
    n_classes = len(classes)
    print(f"Classes ({n_classes}): {classes}")
    print(f"Train: {len(train_loader.dataset):,}  |  Val: {len(val_loader.dataset):,}\n")

    csv_path = f"results_worker{args.worker_id}.csv"

    # Comprova quines configs ja s'han fet (per si s'ha interromput i es repren)
    done_ids = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done_ids = {int(r["config_id"]) for r in csv.DictReader(f)}
        print(f"Configs ja fetes (es saltaran): {sorted(done_ids)}\n")

    t_total = time.time()

    for local_i, (global_id, cfg) in enumerate(zip(global_indices, my_configs)):
        if global_id in done_ids:
            print(f"[{local_i+1}/{len(my_configs)}] Config {global_id} ja feta, saltant.")
            continue

        blocks_str = str(cfg["conv_blocks"])
        print(f"\n{'='*70}")
        print(f"[{local_i+1}/{len(my_configs)}] Config {global_id}")
        print(f"  conv_blocks  : {blocks_str}")
        print(f"  dropout_conv : {cfg['dropout_conv']}")
        print(f"  dropout_fc   : {cfg['dropout_fc']}")
        print(f"{'='*70}")

        val_acc, elapsed, n_params = train_config(
            cfg, train_loader, val_loader, n_classes, device, global_id
        )

        save_result(csv_path, global_id, cfg, val_acc, elapsed, n_params)
        print(f"  → val_acc={val_acc:.2%}  t={elapsed:.0f}s  params={n_params:,}")
        print(f"  Resultat desat a {csv_path}")

    print(f"\n{'='*70}")
    print(f"Worker {args.worker_id} acabat. Temps total: {time.time()-t_total:.0f}s")
    print(f"Resultats a: {csv_path}")
    print(f"\nQuan tots els workers hagin acabat, ajunta els resultats amb:")
    print(f"  python grid_search.py --merge")