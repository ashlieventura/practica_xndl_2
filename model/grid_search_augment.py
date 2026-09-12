# grid_search_augment.py — Cerca d'estratègies d'augmentació de dades
#
# Arquitectura FIXADA: config base [(32,64),(128,256),(256,512)], dropout_conv=0.1,
# dropout_fc=0.3, conv strided al bloc final.
#
# Les augmentacions estan dissenyades per al problema concret:
# classes d'objectes rodons (poma, pilota, brúixola, galeta...) que es distingeixen
# pel SEU INTERIOR, no per la forma exterior. L'objectiu és forçar el model a
# fixar-se en els detalls interns en lloc de la silueta circular.
#
# Ús:
#   python grid_search_augment.py                          # ordinador únic
#   python grid_search_augment.py --n_workers 3 --worker_id 0
#   python grid_search_augment.py --n_workers 3 --worker_id 1
#   python grid_search_augment.py --n_workers 3 --worker_id 2
#   python grid_search_augment.py --merge

import os, time, csv, argparse, random, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image as PILImage
from concurrent.futures import ThreadPoolExecutor


BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ================================================================
# ARQUITECTURA FIXADA — no es varia en aquesta cerca
# ================================================================
FIXED_MODEL = {
    "conv_blocks":   [(32, 64), (128, 256), (256, 512)],
    "dropout_conv":  0.1,
    "dropout_fc":    0.3,
    "use_bn":        True,
    "head":          "gap",
    "fc_sizes":      [256],
}

# Hiperparàmetres d'entrenament fixos
FIXED_TRAIN = {
    "data_dir":          "../dades",
    "batch_size":        512,
    "lr":                2e-3,
    "optimizer":         "adam",
    "weight_decay":      0,
    "scheduler":         "onecycle",
    "reducelr_factor":   0.5,
    "reducelr_patience": 5,
    "label_smooth":      0.1,
    "max_epochs":        29,
    "time_limit":        13 * 60,
    "seed":              0,
    "cache_in_ram":      True,
}

# ================================================================
# ESPAI DE CERCA — només augmentació
# ================================================================
#
# Cada config és un dict amb:
#   name        — identificador llegible per al CSV
#   description — per què esperem que ajudi
#   params      — kwargs que passem a make_transform()
#
AUGMENT_CONFIGS = [
    # {
    #     "name": "none",
    #     "description": "Sense augmentació. Baseline pur.",
    #     "params": {},
    # },
    # {
    #     "name": "light_base",
    #     "description": "Rotació lleugera + translació. Config de referència del cnn_iter.",
    #     "params": {
    #         "rotation":    10,
    #         "translate":   (0.06, 0.06),
    #     },
    # },
    {
        "name": "erasing_small",
        "description": (
            "RandomErasing amb parxos petits (2-10% de l'àrea). "
            "Obliga el model a no dependre d'una zona puntual, però deixa "
            "visible la major part de l'interior de l'objecte."
        ),
        "params": {
            "rotation":      10,
            "translate":     (0.06, 0.06),
            "erasing":       True,
            "erasing_scale": (0.02, 0.10),
            "erasing_ratio": (0.3, 3.0),
            "erasing_p":     0.5,
        },
    },
    {
        "name": "erasing_medium",
        "description": (
            "RandomErasing amb parxos mitjans (8-20% de l'àrea). "
            "Pressió més forta: el model ha de reconèixer l'objecte "
            "amb part del centre esborrat."
        ),
        "params": {
            "rotation":      10,
            "translate":     (0.06, 0.06),
            "erasing":       True,
            "erasing_scale": (0.08, 0.20),
            "erasing_ratio": (0.5, 2.0),
            "erasing_p":     0.5,
        },
    },
    {
        "name": "cutout_center",
        "description": (
            "CutOut centrat: tapa una finestra fixa al centre de la imatge "
            "(on sol estar l'interior de l'objecte rodó). "
            "Força el model a inferir la classe des dels marges/textures."
        ),
        "params": {
            "rotation":    10,
            "translate":   (0.06, 0.06),
            "cutout":      True,
            "cutout_size": 10,   # píxels (sobre 32×32, ~10% de l'àrea)
            "cutout_p":    0.5,
        },
    },
    {
        "name": "cutout_large",
        "description": (
            "CutOut centrat gran (16px, ~25% de l'àrea). "
            "Versió agressiva: tapa la meitat central. "
            "Útil si el model s'ancora massa al centre."
        ),
        "params": {
            "rotation":    10,
            "translate":   (0.06, 0.06),
            "cutout":      True,
            "cutout_size": 16,
            "cutout_p":    0.5,
        },
    },
    {
        "name": "random_crop_zoom",
        "description": (
            "RandomResizedCrop: retalla entre el 60-100% de la imatge i "
            "escala fins a 32×32. Fa zoom en subregions i força el model "
            "a reconèixer l'objecte des d'un fragment parcial."
        ),
        "params": {
            "rotation":          10,
            "random_crop":       True,
            "crop_scale":        (0.60, 1.00),
            "crop_ratio":        (0.85, 1.15),  # quasi quadrat per imatges rodones
        },
    },
    {
        "name": "erasing_and_crop",
        "description": (
            "Combinació de RandomResizedCrop i RandomErasing mitjà. "
            "Doble pressió: el model veu fragments de l'objecte I amb zones esborrades. "
            "La combinació més agressiva de les provades."
        ),
        "params": {
            "rotation":          10,
            "translate":         (0.06, 0.06),
            "random_crop":       True,
            "crop_scale":        (0.65, 1.00),
            "crop_ratio":        (0.85, 1.15),
            "erasing":           True,
            "erasing_scale":     (0.05, 0.15),
            "erasing_ratio":     (0.5, 2.0),
            "erasing_p":         0.4,
        },
    },
    {
        "name": "multi_erase",
        "description": (
            "Dos RandomErasing independents de mida petita (p=0.4 cadascun). "
            "En lloc d'un parxo gran, pot tapar dues zones petites diferents "
            "de l'interior. Menys agressiu però més diversificat espacialment."
        ),
        "params": {
            "rotation":       10,
            "translate":      (0.06, 0.06),
            "multi_erasing":  True,
            "erasing_scale":  (0.02, 0.08),
            "erasing_ratio":  (0.3, 3.0),
            "erasing_p":      0.4,
            "n_erasings":     2,
        },
    },
    {
        "name": "brightness_contrast",
        "description": (
            "ColorJitter (brightness + contrast) sense augmentació espacial. "
            "Prova de control: augmentar la variabilitat d'intensitat ajuda "
            "quan les classes es distingeixen per textura, però aquí la "
            "forma/estructura sembla més rellevant."
        ),
        "params": {
            "rotation":    10,
            "translate":   (0.06, 0.06),
            "jitter":      True,
            "brightness":  0.3,
            "contrast":    0.3,
        },
    },
]


# ================================================================
# TRANSFORMS PERSONALITZADES
# ================================================================

class CutOutTransform:
    """Tapa una finestra quadrada centrada (o aleatòria) amb el valor mitjà.
    Diferent de RandomErasing: el parxo és sempre al centre, que és on
    sol estar l'interior dels objectes rodons d'aquest dataset."""
    def __init__(self, size, p=0.5, centered=True):
        self.size     = size
        self.p        = p
        self.centered = centered

    def __call__(self, img):
        if random.random() > self.p:
            return img
        # Treballem sobre tensor [C, H, W]
        _, h, w = img.shape
        if self.centered:
            cy, cx = h // 2, w // 2
        else:
            cy = random.randint(self.size // 2, h - self.size // 2)
            cx = random.randint(self.size // 2, w - self.size // 2)
        y1 = max(0, cy - self.size // 2)
        y2 = min(h, cy + self.size // 2)
        x1 = max(0, cx - self.size // 2)
        x2 = min(w, cx + self.size // 2)
        # Emplena amb el valor mitjà del canal (0.0 després de normalitzar a [-1,1])
        img[:, y1:y2, x1:x2] = 0.0
        return img


class MultiErasingTransform:
    """Aplica N RandomErasing independents. Permet borrar múltiples zones
    petites en lloc d'una de gran, cobrint millor l'espai interior."""
    def __init__(self, n, scale, ratio, p):
        self.erasings = [
            transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=0)
            for _ in range(n)
        ]

    def __call__(self, img):
        for e in self.erasings:
            img = e(img)
        return img


def make_transform(params):
    """Construeix el pipeline d'augmentació de train a partir d'un dict de paràmetres.
    El pipeline d'eval és sempre el base (sense augmentació)."""

    base_post = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]

    # Transforms geomètriques (operades sobre PIL, abans de ToTensor)
    pre = [transforms.Grayscale(num_output_channels=1)]

    if params.get("random_crop"):
        pre.append(transforms.RandomResizedCrop(
            size=32,
            scale=params.get("crop_scale", (0.7, 1.0)),
            ratio=params.get("crop_ratio", (0.9, 1.1)),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ))
    else:
        # Sense crop: rotació + translació estàndard
        if params.get("rotation", 0) > 0:
            pre.append(transforms.RandomRotation(params["rotation"]))
        if params.get("translate"):
            pre.append(transforms.RandomAffine(degrees=0, translate=params["translate"]))

    if params.get("jitter"):
        pre.append(transforms.ColorJitter(
            brightness=params.get("brightness", 0.2),
            contrast=params.get("contrast", 0.2),
        ))

    # Transforms post-tensor (sobre [C,H,W])
    post = list(base_post)

    if params.get("cutout"):
        post.append(CutOutTransform(
            size=params.get("cutout_size", 10),
            p=params.get("cutout_p", 0.5),
            centered=True,
        ))

    if params.get("multi_erasing"):
        post.append(MultiErasingTransform(
            n=params.get("n_erasings", 2),
            scale=params.get("erasing_scale", (0.02, 0.08)),
            ratio=params.get("erasing_ratio", (0.3, 3.0)),
            p=params.get("erasing_p", 0.4),
        ))
    elif params.get("erasing"):
        post.append(transforms.RandomErasing(
            p=params.get("erasing_p", 0.5),
            scale=params.get("erasing_scale", (0.02, 0.10)),
            ratio=params.get("erasing_ratio", (0.3, 3.0)),
            value=0,
        ))

    train_tf = transforms.Compose(pre + post)
    eval_tf  = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return train_tf, eval_tf


# ================================================================
# DADES
# ================================================================

class InMemoryDataset(torch.utils.data.Dataset):
    """Pre-carrega totes les imatges a RAM amb caché .npy."""
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


def make_loaders(augment_params, device):
    train_tf, eval_tf = make_transform(augment_params)
    data_dir = os.path.join(PROJECT_DIR, FIXED_TRAIN["data_dir"].lstrip("./"))
    train_ds = InMemoryDataset(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds   = InMemoryDataset(os.path.join(data_dir, "val"),   transform=eval_tf)
    pin = (device.type == "cuda")
    nw  = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds, batch_size=FIXED_TRAIN["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False,
        num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0),
    )
    return train_loader, val_loader, train_ds.classes


# ================================================================
# MODEL (arquitectura fixada)
# ================================================================

def build_model(n_classes):
    cfg      = FIXED_MODEL
    n_blocks = len(cfg["conv_blocks"])
    bias     = not cfg["use_bn"]
    c_in     = 1
    layers   = []

    for i, (c_mid, c_out) in enumerate(cfg["conv_blocks"]):
        layers += [nn.Conv2d(c_in, c_mid, 3, padding=1, bias=bias),
                   nn.BatchNorm2d(c_mid), nn.ReLU(inplace=True),
                   nn.Conv2d(c_mid, c_out, 3, padding=1, bias=bias),
                   nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if i < n_blocks - 1:
            layers.append(nn.MaxPool2d(2, 2))
        else:
            # Bloc final: conv strided en lloc de MaxPool
            layers += [nn.Conv2d(c_out, c_out, 3, stride=2, padding=1, bias=bias),
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
        if cfg["dropout_conv"] > 0:
            layers.append(nn.Dropout2d(cfg["dropout_conv"]))
        c_in = c_out

    features = nn.Sequential(*layers)
    c_last   = cfg["conv_blocks"][-1][1]
    head     = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(c_last, cfg["fc_sizes"][0]), nn.ReLU(inplace=True),
        nn.Dropout(cfg["dropout_fc"]),
        nn.Linear(cfg["fc_sizes"][0], n_classes),
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.head     = head
        def forward(self, x): return self.head(self.features(x))

    return Net()


# ================================================================
# ENTRENAMENT
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


def train_config(augment_cfg, device, config_id):
    torch.manual_seed(FIXED_TRAIN["seed"])
    np.random.seed(FIXED_TRAIN["seed"])
    random.seed(FIXED_TRAIN["seed"])

    use_amp   = (device.type == "cuda")
    train_loader, val_loader, classes = make_loaders(augment_cfg["params"], device)
    n_classes = len(classes)
    model     = build_model(n_classes).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)

    criterion = nn.CrossEntropyLoss(label_smoothing=FIXED_TRAIN["label_smooth"])

    decay    = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = optim.Adam(
        [{"params": decay, "weight_decay": FIXED_TRAIN["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=FIXED_TRAIN["lr"],
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=FIXED_TRAIN["lr"],
        steps_per_epoch=len(train_loader),
        epochs=FIXED_TRAIN["max_epochs"],
        pct_start=0.15, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    t_start      = time.time()
    stop         = False

    for epoch in range(1, FIXED_TRAIN["max_epochs"] + 1):
        if stop or time.time() - t_start > FIXED_TRAIN["time_limit"]:
            break

        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for imgs, labels in train_loader:
            if time.time() - t_start > FIXED_TRAIN["time_limit"]:
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
        val_acc, _ = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed    = time.time() - t_start

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"best_augment_cfg{config_id}.pt")

        print(f"  [cfg {config_id} | {augment_cfg['name']:20s}] "
              f"epoch {epoch:>2}  val={val_acc:.2%}  best={best_val_acc:.2%}  t={elapsed:.0f}s")

    return best_val_acc, time.time() - t_start, n_params


# ================================================================
# REPARTIMENT I MERGE
# ================================================================

def assign_configs(all_configs, n_workers, worker_id):
    """Round-robin: cada worker agafa els índexs i % n_workers == worker_id."""
    return [(i, cfg) for i, cfg in enumerate(all_configs) if i % n_workers == worker_id]


def save_result(csv_path, config_id, augment_name, description, val_acc, elapsed, n_params):
    fieldnames = ["config_id", "augment_name", "val_acc", "elapsed_s", "n_params", "description"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "config_id":    config_id,
            "augment_name": augment_name,
            "val_acc":      f"{val_acc:.4f}",
            "elapsed_s":    f"{elapsed:.0f}",
            "n_params":     n_params,
            "description":  description,
        })


def merge_results():
    existing = sorted(
        f for f in os.listdir(".")
        if f.startswith("augment_results_worker") and f.endswith(".csv")
    )
    if not existing:
        print("No s'ha trobat cap fitxer augment_results_workerN.csv")
        return

    rows = []
    for path in existing:
        with open(path) as f:
            rows.extend(list(csv.DictReader(f)))

    rows.sort(key=lambda r: float(r["val_acc"]), reverse=True)

    print(f"\n{'Rank':>4}  {'val_acc':>8}  {'t(s)':>5}  augment_name          description")
    print("─" * 90)
    for rank, r in enumerate(rows, 1):
        desc = r["description"][:50] + ("…" if len(r["description"]) > 50 else "")
        print(f"{rank:>4}  {float(r['val_acc']):>7.2%}  "
              f"{int(float(r['elapsed_s'])):>5}  "
              f"{r['augment_name']:<22}{desc}")

    with open("augment_results_merged.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResultats complets desats a augment_results_merged.csv")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search d'augmentació (arquitectura fixada)"
    )
    parser.add_argument("--n_workers",  type=int, default=1,
                        help="Nombre total d'ordinadors (default: 1)")
    parser.add_argument("--worker_id",  type=int, default=0,
                        help="ID d'aquest ordinador, de 0 a n_workers-1 (default: 0)")
    parser.add_argument("--merge",      action="store_true",
                        help="Ajunta els CSV parcials i imprimeix la taula")
    args = parser.parse_args()

    if args.merge:
        merge_results()
        exit(0)

    assert 0 <= args.worker_id < args.n_workers, \
        f"worker_id ({args.worker_id}) ha de ser entre 0 i n_workers-1 ({args.n_workers-1})"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Worker {args.worker_id} de {args.n_workers}")
    print(f"Arquitectura fixada: {FIXED_MODEL}\n")

    my_configs = assign_configs(AUGMENT_CONFIGS, args.n_workers, args.worker_id)
    print(f"Total de configs d'augmentació: {len(AUGMENT_CONFIGS)}")
    print(f"Configs d'aquest worker: {len(my_configs)} "
          f"(IDs: {[i for i,_ in my_configs]})\n")

    csv_path = f"augment_results_worker{args.worker_id}.csv"

    # Reanudació: salta configs ja fetes
    done_ids = set()
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            done_ids = {int(r["config_id"]) for r in csv.DictReader(f)}
        print(f"Configs ja fetes (es saltaran): {sorted(done_ids)}\n")

    t_total = time.time()

    for local_i, (global_id, aug_cfg) in enumerate(my_configs):
        if global_id in done_ids:
            print(f"[{local_i+1}/{len(my_configs)}] '{aug_cfg['name']}' ja feta, saltant.")
            continue

        print(f"\n{'='*70}")
        print(f"[{local_i+1}/{len(my_configs)}] Config {global_id}: {aug_cfg['name']}")
        print(f"  {aug_cfg['description']}")
        print(f"{'='*70}")

        val_acc, elapsed, n_params = train_config(aug_cfg, device, global_id)

        save_result(csv_path, global_id, aug_cfg["name"], aug_cfg["description"],
                    val_acc, elapsed, n_params)
        print(f"  → val_acc={val_acc:.2%}  t={elapsed:.0f}s")

    print(f"\n{'='*70}")
    print(f"Worker {args.worker_id} acabat. Temps total: {time.time()-t_total:.0f}s")
    print(f"Resultats a: {csv_path}")
    print(f"\nQuan tots els workers hagin acabat:")
    print(f"  python grid_search_augment.py --merge")
