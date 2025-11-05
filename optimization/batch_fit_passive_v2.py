import os
import json
import pandas as pd
from fit_passive_v2 import fit_passive

# === Settings ===
pattern = "iMNTB"  # Change this to filter files by group (e.g., "iMNTB", "WT", or "" for all)
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data","fit_passive","fit_passive"))
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "_fit_passive_results"))

# === Collect matching CSV files ===
all_files = os.listdir(data_dir)
csv_files = [f for f in all_files if f.endswith(".csv") and pattern in f]

print(f"🔍 Found {len(csv_files)} files matching pattern '{pattern}'")

# === Run passive fit for each file ===
summary_paths = []
for fname in csv_files:
    try:
        print(f"\n🔧 Fitting: {fname}")
        passive_params, output_dir = fit_passive(fname)
        print(f"✅ Done: saved to {output_dir}")

        # Find summary file
        json_file = [f for f in os.listdir(output_dir) if f.startswith("passive_summary_") and f.endswith(".json")]
        if json_file:
            summary_paths.append(os.path.join(output_dir, json_file[0]))

    except Exception as e:
        print(f"❌ Error processing {fname}: {e}")

# === Aggregate all JSON summaries into a single CSV ===
if summary_paths:
    print(f"\n📊 Aggregating {len(summary_paths)} passive summary files...")
    records = []
    for path in summary_paths:
        with open(path, "r") as f:
            record = json.load(f)
            fname = os.path.basename(path)

            # Extract filename metadata
            parts = fname.split("_")
            try:
                record["age"] = next(p for p in parts if p.startswith("P"))
                record["group"] = "TeNT" if "TeNT" in parts else ("iMNTB" if "iMNTB" in parts else "WT")
                record["date"] = next((p for p in parts if p.isdigit() and len(p) == 8), "unknown")
                record["cell_id"] = next((p for p in parts if p.startswith("S") and "C" in p), "unknown")
            except Exception:
                record["age"] = record["group"] = record["date"] = record["cell_id"] = "unknown"

            record["source_file"] = fname
            records.append(record)

    df_summary = pd.DataFrame(records)
    summary_csv_path = os.path.join(results_dir, f"passive_fit_summary_{pattern}.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"✅ Saved combined summary CSV to:\n   {summary_csv_path}")
else:
    print("⚠️ No summary files found to aggregate.")
