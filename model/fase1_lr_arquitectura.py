# FASE 1 — Grid search: LEARNING RATE + ARQUITECTURA
# =========================================================================
# Primera fase de la cerca. Busquem la combinació de learning rate i
# arquitectura que millor convergeix dins del límit de temps. La resta
# d'hiperparàmetres es deixen en valors per defecte raonables (FIXED) i
# s'afinaran a les fases 2 i 3.
#
# COM EXECUTAR EN VARIS ORDINADORS:
#   En cada ordinador, canvia NOMÉS la variable SHARD_ID (0, 1, 2, ...).
#   Si tens més o menys màquines, ajusta NUM_SHARDS.
#     Ordinador A:  SHARD_ID = 0
#     Ordinador B:  SHARD_ID = 1
#     Ordinador C:  SHARD_ID = 2
#   Cada màquina escriu els seus propis CSV (…_shard0.csv, …_shard1.csv, …).
#   Quan acabin tots, s'ajunten els CSV i s'ordena per best_val_acc.

from xndl_common import run_grid

# ---------------- SHARDING (canvia per ordinador) ----------------
SHARD_ID   = 0     # <-- 0 al primer ordinador, 1 al segon, 2 al tercer...
NUM_SHARDS = 3     # <-- nombre total d'ordinadors

# ---------------- PRESSUPOST DE TEMPS ----------------
TIME_PER_RUN      = 5 * 60        # tope per run (igual que la restricció final)
MAX_EPOCHS        = 40            # cota d'èpoques (el límit real és el temps)
TOTAL_TIME_BUDGET = 12 * 60 * 60  # tope global per a aquest shard

# ---------------- ESPAI DE CERCA ----------------
# Explorem learning rate i arquitectura (canals, profunditat, doble conv, cap).
GRID = {
    "lr":          [5e-4, 1e-3, 3e-3, 5e-3],
    "channels":    [(32, 64, 128), (64, 128, 256), (32, 64, 128, 256)],
    "double_conv": [True, False],
    "n_blocks_fc": ["gap", "flatten"],
}

# Tot el que NO toquem en aquesta fase (valors per defecte raonables):
FIXED = {
    "seed":         0,          # 1 sola llavor en la fase de cerca
    "use_bn":       True,
    "dropout2d":    0.1,
    "dropout_fc":   0.3,
    "init":         "default",
    "optimizer":    "adamw",
    "scheduler":    "onecycle",
    "weight_decay": 5e-4,
    "label_smooth": 0.0,
    "batch_size":   128,
    "augment":      "lastyear",  # augmentation de l'any passat (fix de moment)
}

if __name__ == "__main__":
    run_grid("fase1_lr_arch", GRID, FIXED,
             shard_id=SHARD_ID, num_shards=NUM_SHARDS,
             time_per_run=TIME_PER_RUN, max_epochs=MAX_EPOCHS,
             total_time_budget=TOTAL_TIME_BUDGET)
