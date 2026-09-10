# Segona entrega de la pràctica
#
# A la primera entrega vam utilitzar una arquitectura VGG-style, és a dir, amb blocs de
# doble conv 3×3 + BN + MaxPool, seguint l'estructura típica de VGG.
#
# Després de provar molt l'arquitectura i exprimir-la al màxim, vam veure que no aconseguíem millors 
# resultats, i seguint el feedback, vam decidir canviar a una arquitectura ResNet-style, amb blocs residuals.
# Estem en un bon punt, però no estem trobant la manera de millorar una mica més els resultats.
#
# Per a aquest disseny modular ens hem centrat en un sol dict CONFIG.
# L'hem fet així perquè, a l'hora de trobar el millor model, evidentment havíem d'anar modificant paràmetres.
# Per això tota decisió que es vulgui poder canviar entre execucions viu com a paràmetre dins
# de CONFIG (l'arquitectura de la ResNet no), i les funcions de més avall simplement la llegeixen i hi
# reaccionen. Això permet comparar variants canviant només una línia, mantenint
# el seed fix per assegurar que la comparació sigui justa.

import os, time, random, threading, subprocess
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image as PILImage

CONFIG = {
    "data_dir":   os.environ.get("DATA_DIR", "../dades"),

    "batch_size": 256,
    # Primer vam provar amb un batch gran (512) perquè les imatges són petites (32×32, 1 canal)
    # i caben moltes a la GPU alhora, i podem aconseguir que es facin més èpoques en el temps donat.
    # Al final l'hem acabat reduint una miqueta (256) perquè no hem detectat molta diferència en el temps,
    # i aquest batch size ens ha donat millors resultats (segurannet perque el batch més petit fa que el model 
    # generalitzi millor i no fagi tant overfitting).

    
    "conv_blocks":  [(32, 64), (128, 256), (256, 512)],  

    # Al final no utilitzem VGG, perque hem decidit canviar a una arquitectura ResNet-style, amb blocs residuals.
    # La arquitectura fixada al codi té molts més paràmetres que la VGG, i per tant és més flexible i capaç d'aprendre patrons més complexos,
    # i és el camí que hem trobat per superar els resultats de la VGG. A més, els blocs residuals ajuden a que el model aprengui millor
    #  i més ràpidament, ja que permeten que el gradient flueixi millor a través de la xarxa.


    "use_bn":       True,
     # BatchNorm gairebé sempre ajuda en xarxes amb prou dades per batch
    # (aquí batch=512, més que suficient per estimar bé mitjana/variància):
    # estabilitza l'entrenament, permet lr més alts i sol generalitzar
    # millor. Per això bias=False a les Conv (veure build_model): el
    # biaix de BN ja fa aquesta funció, posar-ne dos és redundant.


    "dropout_conv": 0,
    # Dropout2d perquè en conv és més efectiu
    # eliminar mapes de característiques sencers en comptes de píxels
    # solts (un píxel aïllat eliminat no fa gairebé res si els veïns
    # són gairebé idèntics). Però al final hem decidit no posar-ne, 
    # ja que no millorva gaire els resultats, i hem centrat la regularització 
    # a la part FC, on sí que hi ha més risc d'overfitting.

    "head":       "gap1",
    # En aquesta segona versió hem canviat a flatten (no GAP), perquè així tenim
    # una xarxa més complexa i mes informació de les imatges, no es perd tanta informació espacial.
    # Clau perque el principal problema que tenien els models que provavem era la capacitat de 
    # diferenciar classes molt similars entre elles (cercle, lluna, patata...), i per tant hem 
    # intentat augmentar la complexitat de la xarxa perque pugui aprendre millor aquestes diferències subtils.


    "fc_sizes":   [128],   
    # Hem augmentat de 256 a 512 la mida de la capa FC, perque així el model té més capacitat d'aprendre patrons 
    # complexos i subtils, i té coherència amb la decisió de utilitzar flatten. 


    "dropout_fc": 0.5,
    # Dropout més alt a la part FC que a la conv (0.4 vs 0) perquè
    # les capes denses tenen molts més paràmetres per neurona de sortida
    # i són les que més tendeixen a memoritzar en comptes de generalitzar.


    "lr":           2e-3,
    # LR relativament alt, pensat per anar de la mà amb OneCycle (que
    # ja gestiona la pujada/baixada) i amb AdamW. 
    # Amb un scheduler que decau, val la
    # pena començar agressiu i deixar que el propi scheduler refini.


    "optimizer":    "adamw",      
    # AdamW en comptes d'Adam perquè desacobla el weight decay de
    # l'actualització del gradient i, com es va explicar a classe, amb Adam el
    # WD es barreja amb els
    # moments adaptatius i acaba regularitzant pitjor.
    # A més, SGD es descarta com a opció per defecte perquè sol
    # necessitar més èpoques i ajust fi de lr per igualar Adam/AdamW en
    # temps d'entrenament curt.


    "weight_decay": 5e-4,
    # Valor moderat: prou per regularitzar sense aixafar la capacitat
    # d'aprendre patrons fins dels traços; vam jugar molt amb aquest paràmetre fins a trobar un d'adequat.
    
    "scheduler":    "onecycle",  
    "reducelr_factor":   0.5,    # ReduceLROnPlateau: divideix lr per aquest factor
    "reducelr_patience": 5,      # ReduceLROnPlateau: èpoques sense millora per activar
    # OneCycle escollit com a per defecte perquè amb un nombre d'èpoques
    # fixat per endavant (max_epochs) i un límit de temps estricte
    # (time_limit), OneCycle treu el màxim rendiment en poques èpoques:
    # puja el lr ràpid a l'inici (escapa mínims dolents/explora) i el
    # baixa suaument cap al final (afina). ReduceLROnPlateau és més
    # "reactiu" i útil si no se sap quantes èpoques calen, però aquí ja
    # tenim un pressupost de temps clar, així que OneCycle hi encaixa
    # millor. 'step' es deixa com a opció senzilla de fallback.
    # Vam provar de treure'l i els resultats eren pitjors, ja que ens quedàvem curts en el nombre d'èpoques
    # i el model no aprenia correctament.

    # Seguint el feedback, hem reduit el temps d'escalfament de OneCycle
    # per aprendre més ràpidament al principi.

    "label_smooth": 0.2,
    # Label smoothing perquè moltes classes són
    # ambigües entre elles i es poden confondre les categories; evitar que el model
    # aprengui a predir amb confiança extrema (logits molt grans)
    # sol millorar la calibració i la generalització en aquest tipus
    # de dades sorolloses/ambigües. 

    "max_epochs":  45,
    "time_limit":    30 * 60 - 15,
    # Límit de temps dur (uns 4min45s) perquè l'entrenament es fa dins
    # d'un pressupost de temps fix (p. ex. una sessió/torn limitat);
    # es resta marge (-15s) per deixar temps a guardar el millor model
    # i tancar net abans de tallar-se a mig pas. max_epochs=25 és un
    # sostre teòric per a OneCycle (necessita saber per endavant el
    # total de passos), però normalment el time_limit talla abans.

    

    "augment": "optimized",
    # Hem modificat una mica l'augmentació de dades per afegir transformacions 
    # mes adequades i agressives per les classes que més es confonen entre elles, i
    #  així millorar la generalització del model.


    "seed": 42,
    # Seed fixa per poder comparar configuracions de manera justa
    # entre iteracions (canviar un hiperparàmetre i no la inicialització).

    "ema_decay": 0.999,  # EMA de pesos; 0.0 = desactivat
    # Ho hem estat provant com a última opció però no hem vist millores.
}

class NpzDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        t0   = time.time()
        data = np.load(path)
        self._imgs   = data["images"]           # (N, H, W) uint8
        self.targets = data["labels"].tolist()
        self.classes = data["classes"].tolist() if "classes" in data else \
                       [str(i) for i in range(int(max(self.targets)) + 1)]
        self.transform = transform
        mb = self._imgs.nbytes // 1024 // 1024
        print(f"  {path}  →  {len(self.targets):,} mostres  ({mb} MB)  en {time.time()-t0:.1f}s")

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
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08), shear=6),
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
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
        ])
    elif augment == "optimized":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            
            # 1. Rotació suau: Segura per a totes les classes
            transforms.RandomRotation(15),
            
            # 2. Afí: Mantenim translació i escala
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=0),
            
            # 3. Flip Horitzontal: Multiplica les dades x2 (En horitzonal només, ja que vertical no té sentit per a la majoria de classes)
            transforms.RandomHorizontalFlip(p=0.5),
            
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            
            # 4. Per intentar millorar l'overfitting i les classes similars
            transforms.RandomErasing(p=0.15, scale=(0.01, 0.05), value=0),
        ])
    else:
        raise ValueError(f"augment desconegut: {augment!r}")
    # eval_tf (base) NO porta augmentació: a validació volem mesurar
    # el rendiment real del model sobre la imatge tal qual, no sobre
    # versions aleatòriament distorsionades.
    return train_tf, transforms.Compose(base)


def make_loaders(cfg, device):
    train_tf, eval_tf = make_transforms(cfg["augment"])
    train_ds = NpzDataset(os.path.join(cfg["data_dir"], "train.npz"), transform=train_tf)
    val_ds   = NpzDataset(os.path.join(cfg["data_dir"], "val.npz"),   transform=eval_tf)
    pin = (device.type == "cuda")
    nw  = 4 if device.type == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pin, drop_last=True,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


def build_model(cfg, n_classes):
    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                 self.shortcut = nn.Sequential(
                     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                     nn.BatchNorm2d(out_channels)
                 )
        def forward(self, x):
            out = torch.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            return torch.relu(out)

    # Nova progressió: més profunda (fins a 512)
    features_layers = [
        nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        
        # Bloc 1: 32x32 -> 16x16
        ResBlock(64, 128),
        nn.MaxPool2d(2, 2),
        
        # Bloc 2: 16x16 -> 8x8
        ResBlock(128, 256),
        nn.MaxPool2d(2, 2),
        
        # Bloc 3: 8x8 -> 4x4
        ResBlock(256, 512),
        nn.MaxPool2d(2, 2)
    ]
    
    features = nn.Sequential(*features_layers)

    with torch.no_grad():
        flat_size = features(torch.zeros(1, 1, 32, 32)).numel() # Sortirà 512 * 4 * 4 = 8192

    if cfg["head"] == "gap1":
        head_layers = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        fc_in = 512
    if cfg["head"] == "gap2":
        head_layers = [nn.AdaptiveAvgPool2d(2), nn.Flatten()]
        fc_in = 512 * 2 * 2
    elif cfg["head"] == "flatten":
        head_layers = [nn.Flatten()]
        fc_in = flat_size

    for fc_out in cfg["fc_sizes"]:
        head_layers.append(nn.Linear(fc_in, fc_out))
        head_layers.append(nn.BatchNorm1d(fc_out))
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
            pct_start=0.10, div_factor=10, final_div_factor=100, # <-- CANVI A 0.10
        )

    if cfg["scheduler"] == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    raise ValueError(f"scheduler desconegut: {cfg['scheduler']!r}")


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        imgs_flipped = TF.hflip(imgs)

        with torch.amp.autocast("cuda", enabled=use_amp):
            out_orig    = model(imgs)
            out_flipped = model(imgs_flipped)

            # Consens en LOGITS (Això és vital)
            out = (out_orig + out_flipped) / 2.0
            loss = criterion(out, labels)

        loss_sum += loss.item() * len(labels)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += len(labels)

    return correct / total, loss_sum / total

@torch.no_grad()
def collect_predictions(model, loader, device, use_amp):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        imgs_flipped = TF.hflip(imgs) 
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            out_orig = model(imgs)
            out_flipped = model(imgs_flipped)
            # Consens en LOGITS
            out = (out_orig + out_flipped) / 2.0 
            
        all_probs.append(torch.softmax(out, dim=1).cpu().numpy())
        all_labels.append(labels.numpy())
        
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return probs.argmax(axis=1), labels, probs


def print_metrics(preds, labels, probs, classes):
    n_classes = len(classes)
    total     = len(labels)

    print(f"\nAccuracy global: {(preds == labels).mean():.2%}")

    for k in [k for k in [3, 5] if k < n_classes]:
        topk_preds   = np.argsort(probs, axis=1)[:, -k:]
        topk_correct = sum(int(labels[i]) in topk_preds[i] for i in range(total))
        print(f"Top-{k} accuracy: {topk_correct/total:.2%}")

    print("\nAccuracy per classe:")
    per_class = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class.append(((preds[mask] == c).mean(), classes[c]))
    for acc_c, name in sorted(per_class):
        bar = "█" * int(acc_c * 25)
        print(f"  {name:>22s}: {acc_c:>6.2%}  {bar}")

    print("\nTop-10 confusions (real → predicció):")
    conf = Counter((int(t), int(p)) for t, p in zip(labels, preds) if t != p)
    for (t, p), cnt in conf.most_common(10):
        pct = cnt / (labels == t).sum()
        print(f"  {classes[t]:>22s} → {classes[p]:<22s}  {cnt:>5}  ({pct:.1%})")


class GPUMonitor(threading.Thread):
    def __init__(self, device_index=0, interval=None):
        super().__init__(daemon=True)
        self._lock      = threading.Lock()
        self._stop_evt  = threading.Event()
        self._window    = []
        self._all       = []
        self.backend    = self._init_backend(device_index)
        if interval is None:
            interval = 0.1 if self.backend == "pynvml" else 0.25
        self.interval = interval

    def _init_backend(self, idx):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            return "pynvml"
        except Exception:
            pass
        try:
            self._smi_idx = str(idx)
            subprocess.run(
                ["nvidia-smi", "-i", self._smi_idx,
                 "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, check=True, timeout=5,
            )
            return "smi"
        except Exception:
            return None

    def _sample(self):
        if self.backend == "pynvml":
            return float(self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        out = subprocess.check_output(
            ["nvidia-smi", "-i", self._smi_idx,
             "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return float(out.decode().strip().splitlines()[0])

    def run(self):
        if self.backend is None:
            return
        while not self._stop_evt.wait(self.interval):
            try:
                u = self._sample()
            except Exception:
                continue
            with self._lock:
                self._window.append(u)
                self._all.append(u)

    def reset(self):
        with self._lock:
            self._window = []

    @staticmethod
    def _summary(samples):
        if not samples:
            return None
        return (sum(samples) / len(samples), min(samples), max(samples), len(samples))

    def window_stats(self):
        with self._lock:
            return self._summary(list(self._window))

    def overall_stats(self):
        with self._lock:
            return self._summary(list(self._all))

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2 * self.interval + 1)
        if self.backend == "pynvml":
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


class EMA:
    """Exponential Moving Average dels pesos del model.
    Manté una còpia shadow dels paràmetres i buffers (inclou BN stats)."""

    def __init__(self, model, decay):
        self.decay  = decay
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply(self, model):
        self._backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self._backup)


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    torch.backends.cudnn.benchmark = True  # tria els kernels CUDA més ràpids per a 32×32
    print(f"Device: {device}\n")

    gpu_index   = torch.cuda.current_device() if device.type == "cuda" else 0
    gpu_monitor = GPUMonitor(device_index=gpu_index)
    gpu_monitor.start()
    if gpu_monitor.backend:
        print(f"Monitoratge GPU actiu (backend={gpu_monitor.backend}, "
              f"mostreig cada {gpu_monitor.interval * 1000:.0f}ms)\n")
    else:
        print("Monitoratge GPU NO disponible (ni pynvml ni nvidia-smi)\n")

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
    ema       = EMA(model, CONFIG["ema_decay"]) if CONFIG["ema_decay"] > 0 else None

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
        gpu_monitor.reset()

        pbar = tqdm(train_loader, desc=f"Època {epoch:>2}", leave=False,
                    unit="batch", dynamic_ncols=True)
        for imgs, labels in pbar:
            if time.time() - t_start > CONFIG["time_limit"]:
                # Check de temps DINS del bucle de batches (no només
                # entre èpoques). Una sola època, pot trigar més que tot el marge restant, així que cal
                # poder tallar a mig camí i no esperar que acabi tota
                # l'època per descobrir que ja s'ha exhaurit el temps.
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
            if ema:
                ema.update(model)

            if scheduler is not None and CONFIG["scheduler"] == "onecycle":
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()


            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        gpu_win = gpu_monitor.window_stats()

        if total == 0:
            break

        train_acc, train_loss = correct / total, loss_sum / total
        if ema:
            ema.apply(model)
        val_acc, val_loss = evaluate(model, val_loader, criterion, device, use_amp)
        if ema:
            ema.restore(model)
        elapsed = time.time() - t_start

        # Scheduler per època: reducelr necessita val_loss, step no necessita res
        if scheduler is not None:
            if CONFIG["scheduler"] == "reducelr":
                scheduler.step(val_loss)
            elif CONFIG["scheduler"] == "step":
                scheduler.step()
            # (onecycle ja s'ha actualitzat per batch més amunt, no
            # se'l torna a tocar aquí)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.0f}s  lr={current_lr:.2e}")
        if gpu_win:
            g_avg, g_min, g_max, g_n = gpu_win
            print(f"        GPU util (entrenament): mitjana {g_avg:5.1f}%  "
                  f"(min {g_min:.0f}% / max {g_max:.0f}%, {g_n} mostres)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = ema.shadow if ema else model.state_dict()
            torch.save(state, "best_model.pt")

    gpu_monitor.stop()
    gpu_all = gpu_monitor.overall_stats()
    if gpu_all:
        g_avg, g_min, g_max, g_n = gpu_all
        print(f"\nGPU util mitjana global: {g_avg:.1f}%  "
              f"(min {g_min:.0f}% / max {g_max:.0f}%, {g_n} mostres)")
        print("  (una mitjana baixa de forma sostinguda suggereix coll d'ampolla "
              "al pipeline de dades / CPU; ~90%+ indica bon ús de la GPU)")

    print(f"\nMillor val accuracy: {best_val_acc:.2%}")
    print(f"Temps total: {time.time() - t_start:.0f}s")

    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", weights_only=True))
    preds, labels_np, probs = collect_predictions(model, val_loader, device, use_amp)
    print_metrics(preds, labels_np, probs, classes)