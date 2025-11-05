import os
import re
import shutil
import pandas as pd
from datetime import datetime

# === Set paths ===
parent_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/"
output_dir = os.path.join(parent_dir, "_latest_iMNTB_TeNT_fits")
os.makedirs(output_dir, exist_ok=True)

# === Pattern to extract group and write datetime from folder ===
folder_pattern = re.compile(r'(iMNTB|TeNT(?:x)?)_.*?_S\d+C\d+_(\d{8})_(\d{6})')
file_pattern_ap = re.compile(r'fit_results_exp_(\d{8})_(\d{6})\.csv')
file_pattern_g = re.compile(r'fit_results_(\d{8})_(\d{6})\.csv')
# === Step 1: Collect all folders with valid info ===
folder_info = []
for folder in os.listdir(parent_dir):
    full_path = os.path.join(parent_dir, folder)
    if not os.path.isdir(full_path):
        continue
    match = folder_pattern.search(folder)
    if match:
        group, date_str, time_str = match.groups()
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        folder_info.append((folder, group, dt))

# === Step 2: Keep only the most recent version per "core experiment" ===
latest_versions = {}
for folder, group, dt in folder_info:
    core = "_".join(folder.split("_")[:-2])  # cut off final _YYYYMMDD_HHMMSS
    if core not in latest_versions or dt > latest_versions[core][2]:
        latest_versions[core] = (folder, group, dt)

df_folders = pd.DataFrame(latest_versions.values(), columns=["folder", "group", "datetime"])
last_5_imntb = df_folders[df_folders["group"] == "iMNTB"].sort_values("datetime", ascending=False).head(5)
last_5_tent  = df_folders[df_folders["group"].str.startswith("TeNT")].sort_values("datetime", ascending=False).head(5)
selected_folders = pd.concat([last_5_imntb, last_5_tent])
# === Step 2.5: Keep only folders that have at least one expected CSV file ===
def folder_has_expected_files(folder_name):
    folder_path = os.path.join(parent_dir, folder_name)
    try:
        files = os.listdir(folder_path)
        for f in files:
            if file_pattern_ap.match(f) or file_pattern_g.match(f):
                return True
    except Exception as e:
        print(f"❌ Error checking folder {folder_name}: {e}")
    return False

# Filter out folders without expected files
selected_folders = selected_folders[selected_folders["folder"].apply(folder_has_expected_files)]

# === Step 3: Clean outdated folders only (skip .py and utility files) ===
existing_items = set(os.listdir(output_dir))
selected_folder_names = set(selected_folders["folder"])

for item in existing_items:
    item_path = os.path.join(output_dir, item)

    # Skip if it's a file (e.g., .py script) or in the latest selected folders
    if item in selected_folder_names:
        continue
    if os.path.isfile(item_path):
        print(f"⚠️ Skipped non-folder file: {item}")
        continue

    # Delete only outdated folders
    try:
        shutil.rmtree(item_path)
        print(f"🗑️ Removed outdated folder: {item}")
    except Exception as e:
        print(f"❌ Failed to delete folder {item}: {e}")


# Copy only if not already copied
for folder_name in selected_folder_names:
    src = os.path.join(parent_dir, folder_name)
    dst = os.path.join(output_dir, folder_name)
    if os.path.exists(dst):
        print(f"📁 Already up-to-date: {dst} — skipping.")
    else:
        shutil.copytree(src, dst)
        print(f"✅ Copied: {folder_name} → _latest_iMNTB_TeNT_fits/")


# === Step 4: Find most recent fit_results_exp_*.csv inside each folder ===
compiled_df = pd.DataFrame()

for _, row in selected_folders.iterrows():
    folder = row["folder"]
    folder_path = os.path.join(output_dir, folder)

    matched_files_ap = []
    matched_files_g = []

    # === Find AP fit files ===
    for f in os.listdir(folder_path):
        match = file_pattern_ap.match(f)
        if match:
            timestamp = "".join(match.groups())
            matched_files_ap.append((f, timestamp))

    # === Find coductance fit files ===
    for f in os.listdir(folder_path):
        match = file_pattern_g.match(f)
        if match:
            timestamp = "".join(match.groups())
            matched_files_g.append((f, timestamp))

    # === Parse AP fit file ===
    if matched_files_ap:
        matched_files_ap.sort(key=lambda x: x[1], reverse=True)
        latest_file_ap = matched_files_ap[0][0]
        file_path_ap = os.path.join(folder_path, latest_file_ap)

        try:
            df_ap = pd.read_csv(file_path_ap)
            df_ap["source_folder"] = folder
            df_ap["group"] = row["group"]
            df_ap["datetime"] = row["datetime"]
            df_ap["fit_file"] = latest_file_ap
            df_ap["fit_type"] = "AP"
            compiled_df = pd.concat([compiled_df, df_ap], ignore_index=True)
            print(f"📄 Parsed AP fit: {latest_file_ap}")
        except Exception as e:
            print(f"❌ Failed to read AP file {file_path_ap}: {e}")
    else:
        print(f"⚠️ No fit_results_exp_*.csv in {folder}")

    # === Parse coductance fit file ===
    if matched_files_g:
        matched_files_g.sort(key=lambda x: x[1], reverse=True)
        latest_file_g = matched_files_g[0][0]
        file_path_g = os.path.join(folder_path, latest_file_g)

        try:
            df_g = pd.read_csv(file_path_g)
            df_g["source_folder"] = folder
            df_g["group"] = row["group"]
            df_g["datetime"] = row["datetime"]
            df_g["fit_file"] = latest_file_g
            df_g["fit_type"] = "coductance"
            compiled_df = pd.concat([compiled_df, df_g], ignore_index=True)
            print(f"📄 Parsed coductance fit: {latest_file_g}")
        except Exception as e:
            print(f"❌ Failed to read coductance file {file_path_g}: {e}")
    else:
        print(f"⚠️ No fit_results_*.csv in {folder}")

# === Step 5: Save compiled CSV into the output folder ===
compiled_ap = compiled_df[compiled_df["fit_type"] == "AP"]
compiled_coductance = compiled_df[compiled_df["fit_type"] == "coductance"]

compiled_ap.to_csv(os.path.join(output_dir, "compiled_fit_results_AP.csv"), index=False)
compiled_coductance.to_csv(os.path.join(output_dir, "compiled_fit_results_coductance.csv"), index=False)

print("✅ Saved separate files:")
print(" - compiled_fit_results_AP.csv")
print(" - compiled_fit_results_coductance.csv")

