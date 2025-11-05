import os
import re
import pandas as pd
from datetime import datetime
import shutil

# === Set the directory where your files are ===
fitted_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_results"
output_dir = os.path.join(fitted_dir, "_latest_all_fitted_params")
os.makedirs(output_dir, exist_ok=True)

# === Regex to extract metadata ===
file_pattern = re.compile(
    r'all_fitted_params_sweep_\d+_clipped_510ms_\d+_P\d+_FVB_PunTeTx_([a-zA-Z]+)_\d+pA_S\d+C\d+_(\d{8})_(\d{6})\.csv'
)

# === Step 1: Collect metadata from files ===
entries = []
for fname in os.listdir(fitted_dir):
    match = file_pattern.match(fname)
    if match:
        group, date_str, time_str = match.groups()
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        entries.append((fname, group, dt))

df = pd.DataFrame(entries, columns=["filename", "group", "datetime"])

# === Step 2: Get latest 5 per group ===
last_5_imntb = df[df["group"].str.lower() == "imntb"].sort_values("datetime", ascending=False).head(5)
last_5_tent  = df[df["group"].str.lower().str.startswith("tent")].sort_values("datetime", ascending=False).head(5)
selected_files = pd.concat([last_5_imntb, last_5_tent])

# === Step 3: Remove outdated files from output folder ===
existing_files = set(os.listdir(output_dir))
selected_file_names = set(selected_files["filename"])

for old_file in existing_files:
    if old_file not in selected_file_names:
        try:
            os.remove(os.path.join(output_dir, old_file))
            print(f"🗑️ Removed outdated file: {old_file}")
        except Exception as e:
            print(f"❌ Failed to delete {old_file}: {e}")

# === Step 4: Copy selected files if not already there ===
for fname in selected_file_names:
    src = os.path.join(fitted_dir, fname)
    dst = os.path.join(output_dir, fname)
    if os.path.exists(dst):
        print(f"📄 Already up-to-date: {fname} — skipping.")
    else:
        shutil.copyfile(src, dst)
        print(f"✅ Copied: {fname}")
