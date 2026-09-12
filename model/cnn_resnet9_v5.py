# Per determinar l'arquitectura base de la nostra xarxa vam cercar quines
# són les millors estructures per a treballs de classificació d'imatges
# 32×32. Evidentment sabíem que volíem utilitzar xarxes convolucionals, ja
# que, com es va veure a classe, és una arquitectura adequada per a la
# classificació d'imatges: aconsegueix que les neurones tinguin menys
# pesos i permet aprendre característiques de les imatges més específiques
# en un menor temps d'entrenament.
#
# En la cerca pel navegador vam trobar aquest repositori, amb una feina
# similar, i ens vam inspirar en la seva arquitectura:
# https://github.com/kradolfer/quickdraw-image-recognition/blob/master/quickdraw_image_recognition.ipynb
#
# És per tant que vam obtenir una base VGG-style, és a dir, amb blocs de
# doble conv 3×3 + BN + MaxPool, seguint l'estructura típica de VGG.
#
# Per què aquest disseny modular ens hem centrat en un sol dict CONFIG.
# L'hem fet així, per que algira de trobar el millor model, evidenment havíem d'anar modificant paràmetres.
# Per això tota decisió que es vulgui poder canviar entre execucions viu com a paràmetre dins
# de CONFIG, i les funcions de més avall simplement la llegeixen i hi
# reaccionen. Això permet comparar variants (p. ex. més blocs vs menys
# dropout vs un altre optimitzador) canviant només una línia, mantenint
# el seed fix per assegurar que la comparació sigui justa.
#
# Els paràmetres es poden agrupar en cinc blocs de decisió, cadascun
# afectant una part diferent del pipeline:

#   - conv_blocks: llista de (c_mid, c_out) — un element = un bloc complet
#     (dues convolucions 3×3 + pooling). Defineix la "forma" de la part
#     convolucional: quants blocs hi ha i quants canals té cadascun.

#   - use_bn, dropout_conv, dropout_fc: controlen la regularització, és
#     a dir, com d'agressivament es força el model a no memoritzar dades
#     d'entrenament i generalitzar millor a validació.

#   - head, fc_sizes: defineixen com es passa de mapes de característiques
#     (sortida convolucional) a les probabilitats finals de classe.
#   - optimizer, scheduler, lr, weight_decay: governen com s'actualitzen
#     els pesos durant l'entrenament (velocitat d'aprenentatge i la seva
#     evolució al llarg de les èpoques).

#   - augment: nivell d'augmentació de dades, és a dir quanta variació
#     artificial (rotacions, translacions...) s'afegeix a les imatges
#     d'entrenament per simular la diversitat real de com la gent dibuixa.

import os, time, random
import threading, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image as PILImage


class GPUMonitor(threading.Thread):
    """Mostreja la utilització de la GPU en un fil de fons (daemon) a un
    ritme fix i constant, independent del bucle d'entrenament.

    Per què un fil separat amb mostreig periòdic i no llegir-ho dins del
    bucle (p. ex. després de cada forward/backward)? Perquè mesurar només
    en moments concrets del pas esbiaixaria la xifra: just després d'un
    backward la GPU sempre està al 100%, just mentre s'espera el següent
    batch (coll d'ampolla de dades) sempre està baixa. Mostrejant a
    intervals regulars amb un rellotge propi capturem la barreja real de
    moments "ocupat" i "esperant dades", de manera que la mitjana reflecteix
    l'ús efectiu de la GPU i delata si el pipeline de dades fa de coll
    d'ampolla (mitjana sostinguda baixa = la GPU espera la CPU/loader).

    nvmlDeviceGetUtilizationRates / `utilization.gpu` retornen el % de temps
    del darrer període de mostreig durant el qual hi havia algun kernel
    executant-se, així que promitjar moltes lectures dóna l'ocupació real.
    """

    def __init__(self, device_index=0, interval=None):
        super().__init__(daemon=True)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._window = []   # mostres des de l'últim reset() (mitjana per època)
        self._all = []      # totes les mostres (mitjana global)
        self.backend = self._init_backend(device_index)
        # pynvml és barat (lectura directa) → podem mostrejar sovint;
        # nvidia-smi llança un subprocés per lectura → interval més ample
        # per no afegir soroll/overhead que pertorbi la pròpia mesura.
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
        # wait() retorna True quan es demana parar; mentre faci timeout
        # (interval) anem mostrejant a ritme constant.
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(PROJECT_DIR, "dades"))

CONFIG = {
    "data_dir":  DATA_DIR,
    "batch_size": 512,
    # Ens hem quedat amb un batch gran (512) perquè les imatges són petites (32×32, 1 canal)
    # i caben moltes a la GPU alhora i podem aconnseguir que es facin més epochs en el temps donat.
    # Si el fem més gran va més lent el model, per el tema del cost computacional i més gran, ha de 
    # fer més batches i per tant triga més també en cada epoch.

    # --- Arquitectura ---
    
    "conv_blocks":  [(32, 64), (128, 256), (256, 512)],  # → 16×16 → 8×8 → 4×4
    # Per què 3 blocs i no més/menys?
    # - 3 blocs porta 32×32 fins a 4×4, que és prou petit perquè el
    #   GAP final resumeixi bé sense perdre massa informació espacial
    #   (amb 4 blocs arribaríem a 2×2, massa agressiu per imatges ja
    #   tan petites i amb poc detall).
    # - Els canals creixen (32→64, 128→256, 256→512) seguint el patró
    #   habitual "menys resolució espacial → més canals", per compensar
    #   la pèrdua d'informació espacial amb més capacitat de
    #   representació per píxel.
    # - El salt 64→128 entre bloc 1 i 2 és el "coll" típic on la xarxa
    #   passa de detectar trets molt locals (vores, traços) a combinar-
    #   los en patrons més abstractes (formes parcials de l'objecte).

    "use_bn":       True,
    # BatchNorm gairebé sempre ajuda en xarxes amb prou dades per batch
    # (aquí batch=512, més que suficient per estimar bé mitjana/variància):
    # estabilitza l'entrenament, permet lr més alts i sol generalitzar
    # millor. Per això bias=False a les Conv (veure build_model): el
    # biaix de BN ja fa aquesta funció, posar-ne dos és redundant.
    "dropout_conv": 0.1,
    # Dropout2d per eliminar mapes de característiques sencers.
    # Valor baix (0.1) per no frenar l'aprenentatge a les primeres epochs.

    # Cap de classificació
    # 'flatten': Flatten → FC → ... → n_classes
    # 'gap':     GlobalAvgPool → FC → n_classes  (menys paràmetres, prova-ho)
    "head":       "gap",
    # GAP en comptes de Flatten perquè:
    # - Amb conv_blocks acabant a 512 canals × 4×4, un Flatten donaria
    #   512*4*4=8192 entrades a la primera FC → moltíssims paràmetres
    #   (8192×256 ≈ 2M només en aquesta capa) i risc alt d'overfitting.
    # - GAP redueix cada mapa de canal a un sol valor (mitjana), assumint
    #   que la presència/intensitat del tret ja indica prou bé la classe,
    #   sense importar gaire la posició exacta — raonable aquí perquè en
    #   un dibuix la forma pot estar una mica descentrada (augmentació
    #   amb translate) i no volem que la xarxa memoritzi posicions fixes.
    # - Com a efecte secundari, fa la xarxa robusta a petites variacions
    #   de mida/posició del dibuix, cosa que encaixa amb com la gent
    #   dibuixa lliurement (no sempre centrat ni a la mateixa escala).
    "fc_sizes":   [256],   # capes FC intermèdies; → n_classes s'afegeix sol
    # Una sola capa oculta de 256 n'hi ha prou: amb GAP l'entrada a la
    # FC ja és compacta (512 valors), no calen diverses capes per
    # "destil·lar" informació; una FC ampla i prou regularitzada sol
    # rendir igual o millor que afegir profunditat aquí.
    "dropout_fc": 0.35,
    # Dropout a la part FC (0.35): lleugerament més alt que a conv.

    # --- Entrenament ---
    "lr":           1.5e-3,
    "optimizer":    "adamw",
    # AdamW desacobla el weight decay dels moments adaptatius,
    # cosa que fa que la regularització sigui més efectiva que amb Adam.
    "weight_decay": 2e-4,
    "scheduler":    "onecycle",  # 'none', 'reducelr', 'onecycle', 'step'
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
    "label_smooth": 0.1,
    "max_epochs": 55,
    # Marge ample: amb una GPU més potent que una 3080 i epochs de ~5s,
    # en 5 min poden cabre 50-60 epochs. El time_limit talla quan toca.
    "time_limit": 30 * 60 - 20,

    # --- Augmentació ---
    # 'none' | 'light' | 'medium' | 'full' | 'erasing_medium'
    "augment": "light",

    # --- Misc ---
    "seed": 42,
    # Seed fixa per poder comparar configuracions de manera justa
    # entre iteracions (canviar un hiperparàmetre i no la inicialització).
    "cache_in_ram": True,  # True: pre-carrega tot a RAM (~144MB), elimina I/O de disc
    # Amb imatges tan petites (32×32, 1 canal) tot el dataset cap
    # còmodament en RAM (~144MB); carregar-ho un cop i servir des de
    # memòria elimina per complet el coll d'ampolla d'I/O de disc en
    # cada època, cosa crítica quan es treballa amb un time_limit ajustat.
}
# ================================================================


class InMemoryDataset(torch.utils.data.Dataset):
    """Pre-carrega totes les imatges a RAM.
    Primera execució: llegeix del disc en paral·lel (threads) i guarda caché .npy.
    Execucions posteriors: carrega el .npy directament (~1-2s fins i tot en HDD)."""
    # Per què construir aquesta classe en comptes d'usar ImageFolder
    # directament (com fa el fallback a make_loaders)?
    # - ImageFolder llegeix del disc a CADA __getitem__, és a dir a
    #   cada època es torna a llegir tot el dataset des de disc. Amb
    #   imatges petites (32×32, 1 canal), el coll d'ampolla real no és
    #   el càlcul sinó l'I/O: moltíssims fitxers petits = moltes
    #   operacions de disc, especialment lent en HDD o discs en xarxa.
    # - Convertint-ho a un únic array NumPy en RAM (~144MB, totalment
    #   assumible) eliminem aquest I/O repetitiu: només es llegeix el
    #   disc un cop (la primera vegada) i la resta d'èpoques són
    #   lectures de memòria, molt més ràpides.
    # - El càlcul en paral·lel amb ThreadPoolExecutor (8 workers) a la
    #   primera càrrega és vàlid perquè PIL.Image.open allibera el GIL
    #   durant la lectura/decodificació, així que els threads sí que
    #   aporten paral·lelisme real aquí (no caldria multiprocessing).
    # - Es guarda una caché .npy a disc perquè rellegir-ho cada vegada
    #   que s'executa l'script (cada iteració d'experiment) seria
    #   malbaratar temps; un .npy es carrega quasi instantàniament
    #   encara que el disc sigui lent, perquè és una sola lectura
    #   seqüencial gran en comptes de milers de lectures petites.
    def __init__(self, folder, transform=None):
        from concurrent.futures import ThreadPoolExecutor
        base = datasets.ImageFolder(folder)
        self.transform = transform
        self.targets   = [s[1] for s in base.samples]
        self.classes   = base.classes
        n = len(base.samples)

        cache = folder.rstrip("/\\") + "_imgs.npz"
        if os.path.exists(cache):
            print(f"  Caché trobada, carregant...", end=" ", flush=True)
            t0 = time.time()
            self._imgs = np.load(cache)["imgs"]
            print(f"fet en {time.time()-t0:.1f}s  ({self._imgs.nbytes//1024//1024}MB)")
        else:
            print(f"  Primera càrrega: {n:,} imatges (es crearà caché .npz per a futures runs)")
            t0 = time.time()
            paths = [p for p, _ in base.samples]

            def _read(p):
                return np.array(PILImage.open(p).convert("L"))

            with ThreadPoolExecutor(max_workers=8) as ex:
                imgs = list(tqdm(ex.map(_read, paths), total=n,
                                unit="img", desc="  Llegint"))
            self._imgs = np.stack(imgs);  del imgs
            np.savez_compressed(cache, imgs=self._imgs)
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
    # Sistema de nivells progressius (none → light → medium → full) en
    # comptes d'una llista fixa de transforms: així es pot anar pujant
    # la intensitat d'augmentació entre iteracions sense haver de
    # reescriure cada vegada quines transformacions s'apliquen, només
    # canviant un sol valor del CONFIG.
    #
    # Decisions comunes a tots els nivells (excepte 'none'):
    # - Mai s'aplica RandomHorizontalFlip a 'light' ni Rotation gran:
    #   un dibuix de Quick, Draw! pot tenir orientació semànticament
    #   rellevant (una fletxa, un número, una lletra) — invertir-lo o
    #   girar-lo massa podria canviar-ne el significat o crear exemples
    #   d'entrenament que no es corresponen amb cap dibuix real.
    # - RandomAffine amb translate petit simula que la persona no
    #   sempre dibuixa centrat al canvas, sense distorsionar la forma.
    # - ColorJitter i RandomErasing només a 'full': tenen poc sentit
    #   teòric en blanc i negre binari (no hi ha "color" real a ajustar)
    #   però es deixen disponibles per si val la pena provar-los com a
    #   regularització extra agressiva en una iteració concreta.
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
    elif augment == "erasing_medium":
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            # RandomErasing amb parxos mitjans (8-20% de l'àrea):
            # obliga el model a reconèixer l'objecte amb part del centre
            # esborrat, clau quan les classes es distingeixen per l'interior.
            transforms.RandomErasing(p=0.5, scale=(0.08, 0.20), ratio=(0.5, 2.0), value=0),
        ])
    else:
        raise ValueError(f"augment desconegut: {augment!r}")
    # eval_tf (base) NO porta augmentació: a validació volem mesurar
    # el rendiment real del model sobre la imatge tal qual, no sobre
    # versions aleatòriament distorsionades.
    return train_tf, transforms.Compose(base)


def make_loaders(cfg, device):
    train_tf, eval_tf = make_transforms(cfg["augment"])
    DS = InMemoryDataset if cfg.get("cache_in_ram", False) else datasets.ImageFolder
    train_ds = DS(os.path.join(cfg["data_dir"], "train"), transform=train_tf)
    val_ds   = DS(os.path.join(cfg["data_dir"], "val"),   transform=eval_tf)
    pin = (device.type == "cuda")
    # Amb cache_in_ram, els workers no fan I/O — serveixen per aplicar transforms.
    # Sense augmentació: 0 workers. Amb augmentació: 2 (més = còpies RAM innecessàries a Windows).
    nw = 4 if device.type == "cuda" else 0
    # Per què condicionar num_workers a si hi ha augmentació?
    # - Sense augmentació, transformar una imatge és gairebé gratuït
    #   (només ToTensor+Normalize), així que llançar processos worker
    #   només afegeix overhead de comunicació (IPC) sense cap guany:
    #   és més ràpid fer-ho tot al procés principal (nw=0).
    # - Amb augmentació (rotacions, affine...), cada transform té un
    #   cost de CPU no menyspreable; repartir-ho en un parell de
    #   workers permet que es vagi preparant el següent batch mentre
    #   la GPU processa l'actual. Es limita a 2 (no més) perquè amb
    #   cache_in_ram cada worker és un procés independent que rep una
    #   còpia/referència de les dades — pujar molt el nombre de workers
    #   incrementa l'ús de memòria i en plataformes com Windows (sense
    #   fork eficient) pot ser fins i tot contraproduent.
    # - pin_memory només té sentit amb GPU (accelera la transferència
    #   CPU→GPU); en CPU pura no aporta res i només consumeix RAM extra.
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=pin,
                              persistent_workers=(nw > 0))
    return train_loader, val_loader, train_ds.classes


def _conv_bn_relu(c_in, c_out, kernel=3, stride=1, padding=1):
    """Bloc bàsic Conv → BN → ReLU. Bias=False perquè BN ja té el seu offset (beta)."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


def build_model(cfg, n_classes):
    """ResNet-9 reduïda per a imatges 32×32.

    Canvis respecte a v4 (9M params, val acc estancada al 86%):
    - Canals reduïts (64→128→128→256 en lloc de 64→128→256→512):
      menys paràmetres (~2.5M vs 9M), menys overfitting, epoch més ràpida.
      Amb 32×32 i 112k imatges no cal tanta capacitat.
    - Skip al bloc 2 afegida: ara tots els blocs tenen residual.
      Amb canals iguals (128→128) no cal projection shortcut.
    - Una sola conv per residual (en lloc de dues): suficient amb
      canals petits i redueix cost computacional.
    - FC(128) en lloc de FC(256): proporcional als canals reduïts.

    Estructura:
      prep  : Conv 1→64                    32×32
      layer1: Conv 64→128  + MaxPool       16×16  + skip(128→128)
      layer2: Conv 128→128 + MaxPool        8×8   + skip(128→128)
      layer3: Conv 128→256 + stride=2       4×4   + skip(256→256)
      cap   : GAP → FC(128) → Dropout → FC(14)
    """
    d_conv = cfg["dropout_conv"]
    d_fc   = cfg["dropout_fc"]

    prep        = _conv_bn_relu(1, 64)

    layer1      = _conv_bn_relu(64, 128)
    layer1_pool = nn.MaxPool2d(2)
    res1        = _conv_bn_relu(128, 128)   # una sola conv
    drop1       = nn.Dropout2d(d_conv)

    layer2      = _conv_bn_relu(128, 128)
    layer2_pool = nn.MaxPool2d(2)
    res2        = _conv_bn_relu(128, 128)   # skip nova al bloc 2
    drop2       = nn.Dropout2d(d_conv)

    layer3      = _conv_bn_relu(128, 256)
    layer3_down = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
    )
    res3        = _conv_bn_relu(256, 256)   # una sola conv
    drop3       = nn.Dropout2d(d_conv)

    gap     = nn.AdaptiveAvgPool2d(1)
    dropout = nn.Dropout(d_fc)
    fc1     = nn.Linear(256, 128)
    fc2     = nn.Linear(128, n_classes)

    class ResNet9(nn.Module):
        def __init__(self):
            super().__init__()
            self.prep        = prep
            self.layer1      = layer1
            self.layer1_pool = layer1_pool
            self.res1        = res1
            self.drop1       = drop1
            self.layer2      = layer2
            self.layer2_pool = layer2_pool
            self.res2        = res2
            self.drop2       = drop2
            self.layer3      = layer3
            self.layer3_down = layer3_down
            self.res3        = res3
            self.drop3       = drop3
            self.gap         = gap
            self.dropout     = dropout
            self.fc1         = fc1
            self.fc2         = fc2

        def forward(self, x):
            x = self.prep(x)

            x = self.layer1(x)
            x = self.layer1_pool(x)
            x = self.drop1(x + self.res1(x))

            x = self.layer2(x)
            x = self.layer2_pool(x)
            x = self.drop2(x + self.res2(x))

            x = self.layer3(x)
            x = self.layer3_down(x)
            x = self.drop3(x + self.res3(x))

            x = self.gap(x).flatten(1)
            x = self.dropout(x)
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return ResNet9()



def build_optimizer(cfg, model):
    lr, wd = cfg["lr"], cfg["weight_decay"]
    # Separem els paràmetres: weight decay només a matrius de pesos (dim >= 2).
    # BN (gamma/beta, dim=1) i biases (dim=1) queden exempts — regularitzar-los
    # perjudica l'entrenament.
    # Per què? El weight decay "estira" els paràmetres cap a zero per
    # evitar pesos massa grans (overfitting). Però gamma/beta de BN i
    # els biases no representen "magnitud de connexió" sinó desplaçaments
    # i escalats puntuals; penalitzar-los cap a zero no millora la
    # generalització i pot, de fet, desestabilitzar BN (per exemple
    # empenyent gamma cap a 0 redueix la capacitat expressiva de la
    # normalització). És una pràctica estàndard (p. ex. recomanada als
    # papers i implementacions de referència de ResNet/EfficientNet).
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
        # pct_start=0.15: només el 15% inicial dels passos es dedica a
        # "escalfar" pujant el lr fins a max_lr; la resta (85%) és
        # baixada cap a un lr molt petit. Amb un dataset gran i un
        # time_limit ajustat, val més gastar la majoria del temps
        # "afinant" (lr baixant) que escalfant llarg.
        # div_factor=10: lr inicial = max_lr/10, prou suau per no
        # divergir als primers passos abans que BN s'hagi estabilitzat.
        # final_div_factor=100: lr final = lr_inicial/100, és a dir
        # molt proper a zero al final del cicle — permet que les
        # últimes èpoques facin ajustos molt fins sense "saltar" del
        # mínim trobat.
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
    # AMP (precisió mixta) només té sentit/suport fiable amb CUDA aquí
    # (torch.amp.autocast("cuda", ...) està fixat a "cuda" més avall).
    # Amb AMP, gran part del càlcul es fa en float16 enlloc de float32,
    # cosa que accelera molt l'entrenament en GPU moderna i redueix
    # memòria — important per poder fer servir batch_size=512 còmodament.
    # GradScaler evita que els gradients petits "desapareguin" (underflow)
    # en float16.
    print(f"Device: {device}\n")

    gpu_index = torch.cuda.current_device() if device.type == "cuda" else 0
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
        gpu_monitor.reset()  # buida la finestra: la mitjana serà només d'aquesta època

        pbar = tqdm(train_loader, desc=f"Època {epoch:>2}", leave=False,
                    unit="batch", dynamic_ncols=True)
        for imgs, labels in pbar:
            if time.time() - t_start > CONFIG["time_limit"]:
                # Check de temps DINS del bucle de batches (no només
                # entre èpoques): amb un dataset gran, una sola època
                # pot trigar més que tot el marge restant, així que cal
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

            if scheduler is not None and CONFIG["scheduler"] == "onecycle":
                if scheduler.last_epoch < scheduler.total_steps - 1:
                    scheduler.step()

            loss_sum += loss.item() * len(labels)
            correct  += (out.argmax(1) == labels).sum().item()
            total    += len(labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.2%}")

        # Capturem la utilització de la fase d'entrenament ABANS d'avaluar,
        # per aïllar l'ús de la GPU als passos d'entrenament (on es nota el
        # coll d'ampolla del loader), sense barrejar-hi la validació.
        gpu_win = gpu_monitor.window_stats()

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
            torch.save(model.state_dict(), "best_model.pt")
            # Es guarda només quan millora el val_acc (no l'últim model
            # ni per cada època): com que el time_limit pot tallar
            # l'entrenament en qualsevol punt —fins i tot just després
            # d'una època dolenta per soroll d'augmentació o un lr
            # encara alt—, volem quedar-nos sempre amb la millor versió
            # vista fins ara, no amb la més recent.

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