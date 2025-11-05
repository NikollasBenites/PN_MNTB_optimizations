import os
import re
import pandas as pd
from datetime import datetime
import shutil
# === Set the path where your files are ===
passive_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_results"

# === Pattern to extract info from filename ===
file_pattern = re.compile(
    r'passive_params_experimental_data_(\d{8})_P\d+_FVB_PunTeTx_([a-zA-Z]+)_\d+pA_S\d+C\d+_.*?_(\d{8})_(\d{6})\.txt'
)

# === Step 1: List all passive fit files and extract metadata ===
entries = []
for fname in os.listdir(passive_dir):
    match = file_pattern.match(fname)
    if match:
        exp_date, group, fit_date, fit_time = match.groups()
        dt = datetime.strptime(fit_date + fit_time, "%Y%m%d%H%M%S")
        entries.append((fname, group, dt))

# === Step 2: Create DataFrame and select top 5 latest per group ===
df = pd.DataFrame(entries, columns=["filename", "group", "datetime"])

# Filter groups
last_5_imntb = df[df["group"].str.lower() == "imntb"].sort_values("datetime", ascending=False).head(5)
last_5_tent  = df[df["group"].str.lower().str.startswith("tent")].sort_values("datetime", ascending=False).head(5)

# === Step 3: Combine and output ===
selected_files = pd.concat([last_5_imntb, last_5_tent])

print("✅ Selected latest 5 files per group:")
print(selected_files[["filename", "group", "datetime"]])

# === Step 4: Copy selected files and clean outdated ones ===
output_dir = os.path.join(passive_dir, "_latest_passive_fits")
os.makedirs(output_dir, exist_ok=True)

# Get currently existing files in the output folder
existing_files = set(os.listdir(output_dir))
selected_file_names = set(selected_files["filename"])

# 🗑️ Delete outdated files not in the current selection
for old_file in existing_files:
    if old_file not in selected_file_names:
        try:
            os.remove(os.path.join(output_dir, old_file))
            print(f"🗑️ Removed outdated file: {old_file}")
        except Exception as e:
            print(f"❌ Failed to delete {old_file}: {e}")

# ✅ Copy new files (skip if already present)
for fname in selected_file_names:
    src = os.path.join(passive_dir, fname)
    dst = os.path.join(output_dir, fname)
    if os.path.exists(dst):
        print(f"📄 Already up-to-date: {fname} — skipping.")
    else:
        shutil.copyfile(src, dst)
        print(f"✅ Copied: {fname}")

