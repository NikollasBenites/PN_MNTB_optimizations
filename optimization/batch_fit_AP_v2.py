import os
import pandas as pd
import re

# === Settings ===
pattern = "TeNT"   # Change this to filter files by group
sweep = "sweep"     # strategy to match files
passive = "passive" # strategy to match files in passive

if pattern == "iMNTB":
    from fit_AP_v2_iMNTB import fit_ap_imntb as fit_ap
    print("Running iMNTB")
    filename_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ap_P9_iMNTB"))
    param_file_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results","_fit_results","_latest_passive_fits","iMNTB"))
elif pattern == "TeNT":
    from fit_AP_v2_TeNT import fit_ap_tent as fit_ap
    print("Running TeNT")
    filename_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ap_P9_TeNT"))
    param_file_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "results", "_fit_results", "_latest_passive_fits", "TeNT"))
else:
   print("Not a valid pattern")

results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "_fit_results", f"batch_fit_AP_{pattern}"))
print(f'Directory created: {results_dir}')
os.makedirs(results_dir, exist_ok=True)

# === Collect matching CSV files ===
all_filenames = os.listdir(filename_dir)
all_param_files = os.listdir(param_file_dir)
csv_files = [f for f in all_filenames if f.endswith(".csv") and pattern and sweep in f]
txt_params = [f for f in all_param_files if f.endswith(".txt") and pattern and passive in f]

def extract_cell_id(filename):
    date_match = re.search(r'(\d{8})', filename)
    cell_match = re.search(r'_(S\d+C\d+)', filename)
    if date_match and cell_match:
        return date_match.group(1), cell_match.group(1)

def extract_stim_amp(filename):
    match = re.search(r'(\d+)pA', filename)
    return float(match.group(1)) / 1000 - 0.02 if match else None

data =[]

for csv in csv_files:
    date, cell_id = extract_cell_id(csv)
    stim_amp = extract_stim_amp(csv)
    if stim_amp is None:
        continue
    match_txt = next((txt for txt in txt_params if date in txt and cell_id in txt), None)
    data.append({
        "date": date,
        "cell_id": cell_id,
        "csv": csv,
        "stim_amp": stim_amp,
        "param_file": match_txt
    })
df = pd.DataFrame(data)

print(f'''🔍 Found {len(csv_files)} files matching pattern '{pattern}'to do AP fitting''')
print("=== AP files ===")
for i, f in enumerate(csv_files, 1):
    print(f"{i}. {f}")

print(f'''🔍 Found {len(txt_params)} files matching pattern '{pattern}'to do with passive params''')
print("=== Passive param files ===")
for i, f in enumerate(txt_params, 1):
    print(f"{i}. {f}")

# === Run ap fit for each file ===
file_paths = []
param_paths = []

for idx, row in df.iterrows():
    try:
        print(f"\n🔧 Fitting: {row['csv']}")
        output_dir = fit_ap(row['csv'], row['stim_amp'], row['param_file'],batch_mode=True, expected_pattern=pattern)
        print(f"✅ Done: saved to {output_dir}")

    except Exception as e:
        print(f"❌ Error processing {row['csv']}: {e}")

