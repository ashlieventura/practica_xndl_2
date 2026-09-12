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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image as PILImage

CONFIG = {
    "data_dir":   "../dades",
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
    "dropout_conv": 0.15,
    # Dropout2d (no Dropout normal) perquè en conv és més efectiu
    # eliminar mapes de característiques sencers en comptes de píxels
    # solts (un píxel aïllat eliminat no fa gairebé res si els veïns
    # són gairebé idèntics). Valor baix (0.15) perquè el dataset és
    # gran (Quick, Draw! té moltíssimes mostres per classe) i el
    # principal risc no és tant overfitting de paràmetres com manca
    # de regularització geomètrica — per això la càrrega forta de
    # regularització recau més en l'augmentació (avall) que aquí.

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
    "dropout_fc": 0.3,
    # Dropout més alt a la part FC que a la conv (0.3 vs 0.15) perquè
    # les capes denses tenen molts més paràmetres per neurona de sortida
    # i són les que més tendeixen a memoritzar en comptes de generalitzar;
    # és el patró habitual: poca regularització a conv, més a FC.

    # --- Entrenament ---
    "lr":           2e-3,
    # LR relativament alt, pensat per anar de la mà amb OneCycle (que
    # ja gestiona la pujada/baixada) i amb AdamW (menys sensible a un
    # lr "agressiu" que SGD pla). Amb un scheduler que decau, val la
    # pena començar agressiu i deixar que el propi scheduler refini.
    "optimizer":    "adamw",      # 'adam', 'adamw', 'sgd'
    # AdamW en comptes d'Adam perquè desacobla el weight decay de
    # l'actualització del gradient (a Adam el WD es barreja amb els
    # moments adaptatius i acaba regularitzant pitjor / de manera menys
    # predictible). SGD es descarta com a opció per defecte perquè sol
    # necessitar més èpoques i ajust fi de lr per igualar Adam/AdamW en
    # temps d'entrenament curt (recordem time_limit ajustat, més avall).
    "weight_decay": 1e-4,
    # Valor moderat: prou per regularitzar sense aixafar la capacitat
    # d'aprendre patrons fins dels traços. Combinat amb el filtre per
    # dimensió a build_optimizer (només es penalitzen pesos, no BN/bias).
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
    # Label smoothing perquè moltes classes de Quick, Draw! són
    # ambigües entre elles (p. ex. dibuixos esquemàtics que es poden
    # confondre entre categories properes); evitar que el model
    # aprengui a predir amb confiança extrema (logits molt grans)
    # sol millorar la calibració i la generalització en aquest tipus
    # de dades sorolloses/ambigües.
    "max_epochs":  28,
    "time_limit":   5 * 60 - 15,
    # Límit de temps dur (uns 4min45s) perquè l'entrenament es fa dins
    # d'un pressupost de temps fix (p. ex. una sessió/torn limitat);
    # es resta marge (-15s) per deixar temps a guardar el millor model
    # i tancar net abans de tallar-se a mig pas. max_epochs=25 és un
    # sostre teòric per a OneCycle (necessita saber per endavant el
    # total de passos), però normalment el time_limit talla abans.

    # --- Augmentació ---
    # 'none' | 'light' | 'medium' | 'full'
    "augment": "light",
    # 'light' (rotació petita + translació petita) com a punt de
    # partida raonable per Quick, Draw!: els dibuixos no solen estar
    # girats del tot ni reflectits (un flip horitzontal podria
    # invertir trets asimètrics rellevants, p. ex. una fletxa o un
    # número), així que augmentacions massa agressives ('medium'/'full'
    # amb flip i color jitter) tenen poc sentit en dibuixos en blanc
    # i negre sense color real. 'light' aporta robustesa a petites
    # variacions de mà sense distorsionar la forma essencial del traç.

    # --- Misc ---
    "seed": 0,
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
    has_aug = cfg.get("augment", "none") != "none"
    nw = (2 if has_aug else 0) if device.type == "cuda" else 0
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


def build_model(cfg, n_classes):
    # Cada bloc: Conv(c_in→c_mid) + BN + ReLU + Conv(c_mid→c_out) + BN + ReLU
    #            + MaxPool2d(2) + Dropout2d
    conv_layers = []
    c_in = 1
    bias = not cfg["use_bn"]  # bias innecessari quan BN ja aprèn el seu offset
    # (veure comentari de use_bn al CONFIG): si BN està actiu, el seu
    # paràmetre beta ja desplaça la sortida, així que el bias de la
    # Conv quedaria anul·lat pel BN immediatament després — estalviem
    # paràmetres traient-lo en aquest cas.
    n_blocks = len(cfg["conv_blocks"])
    for i, (c_mid, c_out) in enumerate(cfg["conv_blocks"]):
        conv_layers.append(nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, bias=bias))
        if cfg["use_bn"]:
            conv_layers.append(nn.BatchNorm2d(c_mid))
        conv_layers.append(nn.ReLU(inplace=True))
        conv_layers.append(nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, bias=bias))
        if cfg["use_bn"]:
            conv_layers.append(nn.BatchNorm2d(c_out))
        conv_layers.append(nn.ReLU(inplace=True))

        # Blocs 1 i 2: MaxPool estàndard (sense paràmetres apresos).
        # Bloc final: conv strided 3×3 en lloc de MaxPool.
        # El downsampling és après (no fix), útil quan la resolució
        # ja és petita (8×8 → 4×4) i perdre activacions pot perjudicar.
        if i < n_blocks - 1:
            conv_layers.append(nn.MaxPool2d(2, 2))
        else:
            conv_layers.append(nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1, bias=bias))
            if cfg["use_bn"]:
                conv_layers.append(nn.BatchNorm2d(c_out))
            conv_layers.append(nn.ReLU(inplace=True))

        if cfg["dropout_conv"] > 0:
            conv_layers.append(nn.Dropout2d(cfg["dropout_conv"]))
        c_in = c_out
    features = nn.Sequential(*conv_layers)

    # Càlcul automàtic de la mida del flatten
    # Es passa un tensor fictici (1,1,32,32) per la xarxa per deduir
    # quantes activacions surten al final, en comptes de calcular-ho
    # a mà (32 / 2^n_blocs)^2 * canals_finals. Així, si es canvia
    # conv_blocks al CONFIG (més/menys blocs, stride diferent...), el
    # codi no peta ni cal actualitzar cap número a mà: és autoadaptatiu.
    with torch.no_grad():
        flat_size = features(torch.zeros(1, 1, 32, 32)).numel()

    head_layers = []
    if cfg["head"] == "gap":
        head_layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        fc_in = c_in
        # Amb GAP la mida d'entrada a la FC només depèn dels canals de
        # sortida (c_in, l'últim c_out del bucle), no de la resolució
        # espacial — per això aquí no s'usa flat_size.
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
                # OneCycle s'actualitza per BATCH, no per època (a
                # diferència de step/reducelr, més avall): la corba de
                # lr d'OneCycle està definida sobre el total de passos
                # d'entrenament, no sobre èpoques senceres, així que cal
                # cridar step() a cada batch perquè segueixi la corba
                # prevista correctament. El check de last_epoch evita
                # un error si s'arriba al final exacte del cicle (per
                # exemple si el time_limit talla just a l'últim pas).

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
            # (onecycle ja s'ha actualitzat per batch més amunt, no
            # se'l torna a tocar aquí)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>5}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
              f"{val_loss:>8.4f}  {val_acc:>6.2%}  {elapsed:>5.0f}s  lr={current_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            # Es guarda només quan millora el val_acc (no l'últim model
            # ni per cada època): com que el time_limit pot tallar
            # l'entrenament en qualsevol punt —fins i tot just després
            # d'una època dolenta per soroll d'augmentació o un lr
            # encara alt—, volem quedar-nos sempre amb la millor versió
            # vista fins ara, no amb la més recent.

    print(f"\nMillor val accuracy: {best_val_acc:.2%}")
    print(f"Temps total: {time.time() - t_start:.0f}s")