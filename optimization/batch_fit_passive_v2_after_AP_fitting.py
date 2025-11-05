import os
import glob
from fit_passive_v2_after_AP_fitting import fit_passive

# === CONFIGURATION ===
data_pattern = "*.csv"
param_prefix = "passive_params_"

data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fit_passive", "fit_passive"))
param_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "_fit_passive_results"))

# === Discover experimental data files ===
data_files = sorted(glob.glob(os.path.join(data_dir, data_pattern)))
print(f"📁 Found {len(data_files)} data files.")

for data_path in data_files:
    base = os.path.basename(data_path)
    stem = base.replace(".csv", "")

    # Match the corresponding parameter file
    matching_param = os.path.join(param_dir, f"{param_prefix}{stem}.csv")
    if not os.path.exists(matching_param):
        print(f"⚠️ Skipping: No param file for {base}")
        continue

    try:
        print(f"\n🔎 Rechecking passive fit for: {base}")
        fit_passive(data_path, matching_param)
    except Exception as e:
        print(f"❌ Failed on {base}: {e}")
