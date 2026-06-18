# xndl_common.py — Maquinària compartida pels grid searches per fases.
# =========================================================================
# Conté tot el que comparteixen els tres scripts de cerca (fase 1, 2 i 3):
#   - càrrega de dades i transformacions (amb augmentation configurable),
#   - construcció del model (arquitectura + inicialització de pesos),
#   - construcció d'optimitzador i scheduler,
#   - bucle d'entrenament d'un run amb límit de temps,
#   - runner del grid amb SHARDING (repartir entre N ordinadors), logging a
#     CSV per època i per run, reanudació i tolerància a errors.
#
# Cada script de fase només ha de: definir el seu GRID i els seus FIXED, i
# cridar run_grid(...). Així no es duplica codi i el comportament és idèntic.

import os, time, csv, json, random, itertools, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ============================ CONFIG GLOBAL ============================
DATA_DIR = "../dades"   # ajusta a l'estructura del clúster (train/ i val/)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


# ============================ DADES ============================
def make_transforms(augment):
    """augment pot ser:
       - False / 'none' : sense augmentation
       - True / 'lastyear' : pipeline EXACTE de la solució de l'any passat
       - 'light' / 'medium' : nivells alternatius (per a la fase d'augmentation)
    """
    base_eval = [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    eval_tf = transforms.Compose(base_eval)

    if augment in (False, "none"):
        train_tf = transforms.Compose(base_eval)

    elif augment in (True, "lastyear"):
        # Data augmentation idèntic al de l'any passat (Sergi & Clàudia)
        train_tf = transforms.Compose([
            transforms.RandomRotation(25),
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    elif augment == "light":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    elif augment == "medium":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    else:
        raise ValueError(f"augment desconegut: {augment!r}")

    return train_tf, eval_tf


# Sondeig del dataset un sol cop: nombre de classes
_probe = datasets.ImageFolder(os.path.join(DATA_DIR, "train"))
N_CLASSES = len(_probe.classes)
CLASS_NAMES = _probe.classes


def get_loaders(batch_size, augment):
    train_tf, eval_tf = make_transforms(augment)
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, len(train_ds), len(val_ds)


# ============================ MODEL ============================
def init_weights(model, scheme):
    """Inicialització dels pesos de convolucions i capes lineals.
       scheme: 'default' (la de PyTorch), 'kaiming' o 'xavier'."""
    if scheme == "default":
        return
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if scheme == "kaiming":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "xavier":
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_model(cfg):
    channels  = cfg["channels"]
    use_bn    = cfg.get("use_bn", True)
    double    = cfg.get("double_conv", True)
    d2d       = cfg.get("dropout2d", 0.1)
    dfc       = cfg.get("dropout_fc", 0.3)
    head_type = cfg.get("n_blocks_fc", "gap")
    init      = cfg.get("init", "default")

    def conv_block(c_in, c_out):
        layers = [nn.Conv2d(c_in, c_out, 3, padding=1, bias=not use_bn)]
        if use_bn: layers.append(nn.BatchNorm2d(c_out))
        layers.append(nn.ReLU(inplace=True))
        if double:
            layers.append(nn.Conv2d(c_out, c_out, 3, padding=1, bias=not use_bn))
            if use_bn: layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        if d2d > 0: layers.append(nn.Dropout2d(d2d))
        return nn.Sequential(*layers)

    feats, c_prev = [], 1
    for c in channels:
        feats.append(conv_block(c_prev, c)); c_prev = c
    features = nn.Sequential(*feats)

    spatial = max(32 // (2 ** len(channels)), 1)

    if head_type == "gap":
        head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dfc), nn.Linear(c_prev, N_CLASSES),
        )
    else:  # flatten + dues lineals
        flat = c_prev * spatial * spatial
        head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256),
            nn.BatchNorm1d(256) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True), nn.Dropout(dfc),
            nn.Linear(256, N_CLASSES),
        )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.head = head
        def forward(self, x):
            return self.head(self.features(x))

    net = Net()
    init_weights(net, init)
    return net


def build_optimizer(cfg, params):
    opt = cfg.get("optimizer", "adamw")
    lr  = cfg["lr"]
    wd  = cfg.get("weight_decay", 5e-4)
    if opt == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if opt == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)


# ============================ ENTRENAMENT ============================
def set_seed(seed):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, use_amp):
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


def run_one(cfg, run_id, epochs_writer, epochs_fh, time_per_run, max_epochs):
    set_seed(cfg.get("seed", 0))
    train_loader, val_loader, n_train, n_val = get_loaders(cfg["batch_size"], cfg["augment"])
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smooth", 0.0))
    optimizer = build_optimizer(cfg, model.parameters())

    steps_per_epoch = max(1, len(train_loader))
    scheduler = None
    if cfg.get("scheduler", "onecycle") == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg["lr"], steps_per_epoch=steps_per_epoch,
            epochs=max_epochs, pct_start=0.15)

    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_acc, best_epoch = 0.0, 0
    last = dict(train_acc=float("nan"), train_loss=float("nan"),
                val_acc=float("nan"), val_loss=float("nan"))
    epochs_done = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        if time.time() - t0 > time_per_run:
            break
        model.train()
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            if time.time() - t0 > time_per_run:
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
            if scheduler is not None and scheduler.last_epoch < scheduler.total_steps - 1:
                scheduler.step()
            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
        if total == 0:
            break
        train_acc, train_loss = correct / total, loss_sum / total
        val_acc, val_loss = evaluate(model, val_loader, criterion, use_amp)
        elapsed = time.time() - t0
        epochs_done = epoch
        last.update(train_acc=train_acc, train_loss=train_loss,
                    val_acc=val_acc, val_loss=val_loss)

        epochs_writer.writerow({
            "run_id": run_id, "epoch": epoch,
            "train_loss": f"{train_loss:.5f}", "train_acc": f"{train_acc:.5f}",
            "val_loss": f"{val_loss:.5f}", "val_acc": f"{val_acc:.5f}",
            "elapsed_s": f"{elapsed:.1f}",
        })
        epochs_fh.flush()

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch

    return {
        "n_params": n_params, "n_train": n_train, "n_val": n_val,
        "epochs_done": epochs_done, "best_val_acc": best_val_acc,
        "best_epoch": best_epoch, "final_train_acc": last["train_acc"],
        "final_train_loss": last["train_loss"], "final_val_acc": last["val_acc"],
        "final_val_loss": last["val_loss"], "train_time_s": time.time() - t0,
    }


# ============================ RUNNER DEL GRID (amb sharding) ============================
def cfg_key(cfg):
    return json.dumps(cfg, sort_keys=True, default=str)


def load_done_keys(runs_csv):
    done = set()
    if os.path.exists(runs_csv):
        with open(runs_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("config_json"):
                    done.add(row["config_json"])
    return done


def run_grid(phase_name, GRID, FIXED, *,
             shard_id, num_shards,
             time_per_run, max_epochs, total_time_budget):
    """Executa el grid GRID x FIXED, però només la part que toca a aquest shard.
       Reparteix per índice: shard_id processa els combos amb idx % num_shards == shard_id.
       Escriu a CSV propis del shard perquè cada ordinador no es trepitgi."""

    runs_csv   = f"results_{phase_name}_runs_shard{shard_id}.csv"
    epochs_csv = f"results_{phase_name}_epochs_shard{shard_id}.csv"

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    # combos que toquen a aquest shard
    my_combos = [(i, c) for i, c in enumerate(combos) if i % num_shards == shard_id]

    print(f"=== FASE: {phase_name} ===")
    print(f"Device: {device}")
    print(f"Shard {shard_id+1}/{num_shards}")
    print(f"Combinacions totals: {len(combos)} | en aquest shard: {len(my_combos)}")
    print(f"time_per_run={time_per_run}s  max_epochs={max_epochs}  "
          f"total_budget={total_time_budget}s\n")

    done_keys = load_done_keys(runs_csv)
    if done_keys:
        print(f"Reprenent: {len(done_keys)} runs ja fets en aquest shard es saltaran.\n")

    all_param_keys = keys + list(FIXED.keys())
    runs_fields = ["run_id", "timestamp", "status"] + all_param_keys + [
        "n_params", "n_train", "n_val", "epochs_done", "best_val_acc",
        "best_epoch", "final_train_acc", "final_train_loss",
        "final_val_acc", "final_val_loss", "train_time_s", "config_json", "error"]
    epochs_fields = ["run_id", "epoch", "train_loss", "train_acc",
                     "val_loss", "val_acc", "elapsed_s"]

    runs_exists = os.path.exists(runs_csv)
    epochs_exists = os.path.exists(epochs_csv)
    runs_fh = open(runs_csv, "a", newline="")
    epochs_fh = open(epochs_csv, "a", newline="")
    runs_writer = csv.DictWriter(runs_fh, fieldnames=runs_fields)
    epochs_writer = csv.DictWriter(epochs_fh, fieldnames=epochs_fields)
    if not runs_exists: runs_writer.writeheader(); runs_fh.flush()
    if not epochs_exists: epochs_writer.writeheader(); epochs_fh.flush()

    t_grid = time.time()
    run_id = len(done_keys)
    best_overall = (0.0, None)

    for idx, combo in my_combos:
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED)
        key = cfg_key(cfg)
        if key in done_keys:
            continue
        if time.time() - t_grid > total_time_budget:
            print("\nPressupost total exhaurit. Aturant aquest shard.")
            break

        run_id += 1
        print(f"[{phase_name} s{shard_id} run {run_id}] {cfg}")
        row = {"run_id": run_id, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "config_json": key, "error": ""}
        row.update(cfg)
        try:
            res = run_one(cfg, run_id, epochs_writer, epochs_fh,
                          time_per_run, max_epochs)
            row.update(res); row["status"] = "ok"
            print(f"    -> best_val_acc={res['best_val_acc']:.4f} "
                  f"(epoch {res['best_epoch']}, {res['epochs_done']} èpoques, "
                  f"{res['train_time_s']:.0f}s)")
            if res["best_val_acc"] > best_overall[0]:
                best_overall = (res["best_val_acc"], cfg)
        except Exception as e:
            row["status"] = "failed"; row["error"] = repr(e)
            print(f"    -> FALLAT: {e}")
            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        runs_writer.writerow(row); runs_fh.flush()
        done_keys.add(key)

    runs_fh.close(); epochs_fh.close()
    print("\n================ RESUM SHARD ================")
    if best_overall[1] is not None:
        print(f"Millor val accuracy (aquest shard): {best_overall[0]:.4f}")
        print(f"Millor config: {best_overall[1]}")
    print(f"Resultats per època -> {epochs_csv}")
    print(f"Resultats per run   -> {runs_csv}")
    print("\nQuan tots els shards acabin, ajunta els CSV 'runs' de cada ordinador")
    print("i ordena per best_val_acc per triar la millor config.")
