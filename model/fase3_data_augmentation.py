# FASE 3 — Grid search: DATA AUGMENTATION
# =========================================================================
# Tercera i última fase de cerca. Amb l'arquitectura, el learning rate i la
# regularització ja triats (fases 1 i 2), explorem el data augmentation, que
# és el que més depèn de tota la resta.
#
# Nivells d'augmentation disponibles (definits a xndl_common.make_transforms):
#   - "none"     : sense augmentation
#   - "light"    : rotació petita + afí suau
#   - "medium"   : rotació + flip horitzontal + afí
#   - "lastyear" : el pipeline complet de l'any passat (agressiu)
#
# Aquí, a més, afegim la LLAVOR al grid per començar a mirar la variància:
# així, per a cada nivell d'augmentation, veiem com de estable és el resultat.
#
# IMPORTANT: omple els guanyadors de les fases 1 i 2 abans d'executar.
#
# COM EXECUTAR EN VARIS ORDINADORS: igual que abans, canvia SHARD_ID.

from xndl_common import run_grid

# ---------------- SHARDING (canvia per ordinador) ----------------
SHARD_ID   = 0
NUM_SHARDS = 3

# ---------------- PRESSUPOST DE TEMPS ----------------
TIME_PER_RUN      = 5 * 60
MAX_EPOCHS        = 40
TOTAL_TIME_BUDGET = 12 * 60 * 60

# ---------------- GUANYADORS DE LES FASES 1 i 2 (omple'ls!) ----------------
BEST_CHANNELS    = (64, 128, 256)
BEST_DOUBLE_CONV = True
BEST_HEAD        = "gap"
BEST_LR          = 3e-3
BEST_INIT        = "kaiming"
BEST_DROPOUT2D   = 0.1
BEST_DROPOUT_FC  = 0.3
BEST_WD          = 5e-4
BEST_LABEL_SM    = 0.05
BEST_USE_BN      = True

# ---------------- ESPAI DE CERCA ----------------
GRID = {
    "augment": ["none", "light", "medium", "lastyear"],
    "seed":    [0, 1, 2],   # llavors per veure la variància de cada nivell
}

FIXED = {
    "channels":     BEST_CHANNELS,
    "double_conv":  BEST_DOUBLE_CONV,
    "n_blocks_fc":  BEST_HEAD,
    "lr":           BEST_LR,
    "init":         BEST_INIT,
    "dropout2d":    BEST_DROPOUT2D,
    "dropout_fc":   BEST_DROPOUT_FC,
    "weight_decay": BEST_WD,
    "label_smooth": BEST_LABEL_SM,
    "use_bn":       BEST_USE_BN,
    "optimizer":    "adamw",
    "scheduler":    "onecycle",
    "batch_size":   128,
}

if __name__ == "__main__":
    run_grid("fase3_augmentation", GRID, FIXED,
             shard_id=SHARD_ID, num_shards=NUM_SHARDS,
             time_per_run=TIME_PER_RUN, max_epochs=MAX_EPOCHS,
             total_time_budget=TOTAL_TIME_BUDGET)
