import pandas as pd

df = pd.read_pickle("runs/deCIFer_cifs_v1_model/eval/cifs_v1.pkl.gz")
validity_rate = df["validity"].mean()
rmsd_stats = df["rmsd"].describe()
diffraction_stats = df[["rwp", "wd"]].describe()
spacegroup_match = (df["spacegroup_num_sample"] == df["spacegroup_num_gen"]).mean()

# --- Affichage lisible ---
print("\n=== Résultats de l'évaluation ===")
print(f"✅ Validity rate         : {validity_rate:.3f}")
print(f"✅ Spacegroup match rate : {spacegroup_match:.3f}\n")

print("📊 RMSD stats:")
print(rmsd_stats.to_string(float_format="%.3f"))
print("\n📊 Diffraction stats (Rwp & Wd):")
print(diffraction_stats.to_string(float_format="%.3f"))