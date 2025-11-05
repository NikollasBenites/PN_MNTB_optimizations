import os
import pandas as pd
import re


def parse_simulation_meta(filepath):
    meta_dict = {}
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                meta_dict[key.strip()] = value.strip()
    return meta_dict


def clean_conductance_values(df, conductance_cols):
    for col in conductance_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r"([-\d.]+)").astype(float)
    return df


def gather_simulation_summaries(base_dir):
    summaries = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "simulation_meta.txt":
                filepath = os.path.join(root, file)
                try:
                    meta = parse_simulation_meta(filepath)
                    meta["Folder"] = os.path.basename(root)
                    summaries.append(meta)
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")

    df = pd.DataFrame(summaries)

    # Clean conductances (remove 'nS' and convert to float)
    conductance_cols = ["gLeak", "gNa", "gKHT", "gKLT", "gIH", "gKA","ELeak", "cam", "cbm", "kam","kbm"]
    df = clean_conductance_values(df, conductance_cols)

    return df


# Example usage
if __name__ == "__main__":
    base_directory = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/PN_MNTB_modeling/results/_last_simulations/iMNTB"  # Change to your actual path
    summary_df = gather_simulation_summaries(base_directory)
    summary_df.to_csv("simulation_summary_cleaned_iMNTB.csv", index=False)
    print(summary_df.head())
