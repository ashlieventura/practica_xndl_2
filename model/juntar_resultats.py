# juntar_resultats.py — Ajunta els CSV de tots els shards d'una fase i
# mostra el rànquing de configuracions per best_val_acc.
#
# Ús:
#   python juntar_resultats.py fase1_lr_arch
#   python juntar_resultats.py fase2_init_reg
#   python juntar_resultats.py fase3_augmentation
#
# Busca tots els fitxers results_<fase>_runs_shard*.csv del directori actual,
# els concatena, ordena per best_val_acc i escriu results_<fase>_TOTAL.csv.

import sys, glob
import pandas as pd

if len(sys.argv) < 2:
    print("Ús: python juntar_resultats.py <nom_fase>")
    print("Exemple: python juntar_resultats.py fase1_lr_arch")
    sys.exit(1)

phase = sys.argv[1]
pattern = f"results_{phase}_runs_shard*.csv"
files = sorted(glob.glob(pattern))

if not files:
    print(f"No s'ha trobat cap fitxer amb el patró: {pattern}")
    sys.exit(1)

print(f"Ajuntant {len(files)} fitxers: {files}")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Només runs correctes per al rànquing
ok = df[df["status"] == "ok"].copy()
ok["best_val_acc"] = pd.to_numeric(ok["best_val_acc"], errors="coerce")
ok = ok.sort_values("best_val_acc", ascending=False)

out = f"results_{phase}_TOTAL.csv"
df.to_csv(out, index=False)
print(f"\nGuardat: {out}  ({len(df)} runs totals, {len(ok)} ok, "
      f"{len(df)-len(ok)} fallats)\n")

# Mostra el top 10
cols_show = [c for c in ok.columns if c not in
             ("config_json", "timestamp", "error", "n_train", "n_val")]
print("===== TOP 10 configuracions =====")
with pd.option_context("display.max_columns", None, "display.width", 200):
    print(ok[cols_show].head(10).to_string(index=False))
