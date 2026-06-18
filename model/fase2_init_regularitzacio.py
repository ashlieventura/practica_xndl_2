# FASE 2 — Grid search: INICIALITZACIÓ + REGULARITZACIÓ + DROPOUT
# =========================================================================
# Segona fase. Amb l'ARQUITECTURA i el LEARNING RATE ja triats a la fase 1,
# els fixem aquí i explorem com inicialitzar els pesos i com regularitzar:
#   - init: inicialització dels pesos (default / kaiming / xavier)
#   - dropout2d (a les convolucions) i dropout_fc (a la cap)
#   - weight_decay
#   - label_smooth
#   - use_bn (per confirmar que BatchNorm ajuda en aquesta arquitectura)
#
# IMPORTANT: omple CHANNELS, DOUBLE_CONV, HEAD i LR amb els guanyadors de la
# fase 1 abans d'executar.
#
# COM EXECUTAR EN VARIS ORDINADORS: igual que la fase 1, canvia SHARD_ID.

from xndl_common import run_grid

# ---------------- SHARDING (canvia per ordinador) ----------------
SHARD_ID   = 0
NUM_SHARDS = 3

# ---------------- PRESSUPOST DE TEMPS ----------------
TIME_PER_RUN      = 5 * 60
MAX_EPOCHS        = 40
TOTAL_TIME_BUDGET = 12 * 60 * 60

# ---------------- GUANYADORS DE LA FASE 1 (omple'ls!) ----------------
# Substitueix aquests valors pels millors trobats a la fase 1.
BEST_CHANNELS    = (64, 128, 256)
BEST_DOUBLE_CONV = True
BEST_HEAD        = "gap"
BEST_LR          = 3e-3

# ---------------- ESPAI DE CERCA ----------------
# Mantenim els eixos que de debò cal afinar. (init i use_bn s'han mogut a una
# comprovació ràpida a part, perquè amb BatchNorm la inicialització rarament
# canvia el resultat i barrejar-ho tot multiplicava massa els runs.)
GRID = {
    "dropout2d":    [0.0, 0.1, 0.2],
    "dropout_fc":   [0.0, 0.3, 0.5],
    "weight_decay": [0.0, 1e-4, 5e-4],
    "label_smooth": [0.0, 0.05, 0.1],
}

# Fixos: arquitectura i lr (de la fase 1) + la resta
FIXED = {
    "channels":     BEST_CHANNELS,
    "double_conv":  BEST_DOUBLE_CONV,
    "n_blocks_fc":  BEST_HEAD,
    "lr":           BEST_LR,
    "init":         "kaiming",   # bona opció per defecte amb ReLU
    "use_bn":       True,
    "seed":         0,
    "optimizer":    "adamw",
    "scheduler":    "onecycle",
    "batch_size":   128,
    "augment":      "lastyear",
}

if __name__ == "__main__":
    run_grid("fase2_init_reg", GRID, FIXED,
             shard_id=SHARD_ID, num_shards=NUM_SHARDS,
             time_per_run=TIME_PER_RUN, max_epochs=MAX_EPOCHS,
             total_time_budget=TOTAL_TIME_BUDGET)
