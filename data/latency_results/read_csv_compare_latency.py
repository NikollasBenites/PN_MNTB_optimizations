import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Set the folder path
folder_path = os.path.dirname(os.path.abspath(__file__))

# Target value to search for
target_stim = 240

# === Collect matching data ===
summary_rows = []

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)

        if "Stimulus (pA)" in df.columns:
            # Add phenotype from filename
            phenotype = "TeNT" if "TeNT" in filename else "iMNTB" if "iMNTB" in filename else "Unknown"
            df["Phenotype"] = phenotype
            df["Source File"] = filename
            summary_rows.append(df)

# Combine all rows into one DataFrame
filter = 240
summary_df = pd.concat(summary_rows, ignore_index=True)
clean_df = summary_df[summary_df["Stimulus (pA)"] % 20 == 0]
filtered_df = clean_df[clean_df["Stimulus (pA)"] >= filter]
target_df = filtered_df[filtered_df["Stimulus (pA)"] == target_stim]

# Set style
sns.set(style="whitegrid")

# Define custom color palette
custom_palette = {"iMNTB": "grey", "TeNT": "red"}


plt.figure(figsize=(6, 5))

# Bar plot
sns.barplot(
    data=target_df,
    x="Phenotype",
    y="Latency to Threshold (ms)",
    hue="Phenotype",
    palette=custom_palette,
    errorbar="se",         # use SE for error bars
    capsize=0.2,
    estimator="mean",
    legend=False
)

# Individual points
sns.stripplot(
    data=target_df,
    x="Phenotype",
    y="Latency to Threshold (ms)",
    hue="Phenotype",
    palette="gray",
    size=8,
    jitter=True,
    dodge=False,
    alpha=0.7,
    legend=False
)
plt.ylim([0, 4])  # Set Y-axis (latency) from 0 to 5 ms
plt.title("Latency to Threshold by Phenotype")
plt.ylabel("Latency (ms)")
plt.xlabel("Phenotype")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))

# Bar plot
sns.barplot(
    data=target_df,
    x="Phenotype",
    y="Latency to Peak (ms)",
    hue="Phenotype",
    palette=custom_palette,
    errorbar="se",         # use SE for error bars
    capsize=0.2,
    estimator="mean",
    legend=False
)

# Individual points
sns.stripplot(
    data=target_df,
    x="Phenotype",
    y="Latency to Peak (ms)",
    hue="Phenotype",
    palette="gray",
    size=8,
    jitter=True,
    dodge=False,
    alpha=0.7,
    legend=False
)
plt.ylim([0, 4])  # Set Y-axis (latency) from 0 to 5 ms
plt.title("Latency to Peak by Phenotype")
plt.ylabel("Latency (ms)")
plt.xlabel("Phenotype")
plt.tight_layout()
plt.show()


sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
# --- Plot all individual latency points connected by lines ---
# sns.lineplot(
#     data=filtered_df,
#     x="Stimulus (pA)",
#     y="Latency to Threshold (ms)",
#     hue="Phenotype",
#     units="Source File",     # one line per file/recording
#     estimator=None,          # show all raw values
#     palette=custom_palette,
#     alpha=0.5,
#     linewidth=1
# )

# Optional: overlay group mean
sns.lineplot(
    data=filtered_df,
    x="Stimulus (pA)",
    y="Latency to Threshold (ms)",
    hue="Phenotype",
    palette=custom_palette,
    estimator="mean",
    errorbar="se",
    linewidth=2,
    linestyle="-",
    marker="o",
    legend=True  # hide duplicate legend
)
plt.ylim([0, 4])  # Set Y-axis (latency) from 0 to 5 ms
plt.title(f"Latency to Threshold vs Stimulus (Filtered ≥ {filter} pA)")
plt.xlabel("Stimulus (pA)")
plt.ylabel("Latency (ms)")
plt.legend(title="Phenotype")
plt.tight_layout()
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
# --- Plot all individual latency points connected by lines ---
# sns.lineplot(
#     data=filtered_df,
#     x="Stimulus (pA)",
#     y="Latency to Peak (ms)",
#     hue="Phenotype",
#     units="Source File",     # one line per file/recording
#     estimator=None,          # show all raw values
#     palette=custom_palette,
#     alpha=0.5,
#     linewidth=1
# )

# Optional: overlay group mean
sns.lineplot(
    data=filtered_df,
    x="Stimulus (pA)",
    y="Latency to Peak (ms)",
    hue="Phenotype",
    palette=custom_palette,
    estimator="mean",
    errorbar="se",
    linewidth=2,
    linestyle="-",
    marker="o",
    legend=True  # hide duplicate legend
)
plt.ylim([0, 4])  # Set Y-axis (latency) from 0 to 5 ms
plt.title(f"Latency to Peak vs Stimulus (Filtered ≥ {filter} pA)")
plt.xlabel("Stimulus (pA)")
plt.ylabel("Latency (ms)")
plt.legend(title="Phenotype")
plt.tight_layout()
plt.show()