import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import shapiro, ttest_ind, mannwhitneyu

rcParams['pdf.fonttype'] = 42   # TrueType
# === Set style ===
sns.set_theme(style="whitegrid")

# === Get the directory where the script is located ===
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_dir = os.path.join(script_dir,"..","..","CSV")

# === Create a subfolder for results ===
output_dir = os.path.join(script_dir, "..","..","figures","feature_barplots")
os.makedirs(output_dir, exist_ok=True)

# === Prepare to store all data ===
all_data = []

# === Loop through all CSV files in the CSV ===
for filename in os.listdir(csv_dir):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_dir, filename)
        df = pd.read_csv(file_path)

        # Infer group from filename
        if "iMNTB" in filename:
            df["group"] = "iMNTB"
        elif "TeNT" in filename:
            df["group"] = "TeNT"
        else:
            continue  # Skip if group not identifiable

        df["source_file"] = filename
        all_data.append(df)

# === Combine all data into one DataFrame ===
combined_df = pd.concat(all_data, ignore_index=True)

# === Save the combined data ===
compiled_path = os.path.join(output_dir, "compiled_data.csv")
combined_df.to_csv(compiled_path, index=False)
print(f"✅ Compiled CSV saved at: {compiled_path}")

# === Define custom colors ===
custom_palette = {
    "iMNTB": "#4d4d4d",  # dark grey
    "TeNT": "#fca4a4"  # light red
}

# === Plot barplots with points and statistical annotations ===
numeric_cols = combined_df.select_dtypes(include="number").columns

# === Define custom y-limits for specific parameters ===
ylim_dict = {
    "gna": (0, 350),
    "gkht": (0, 350),
    "gka": (0, 350),
    "gklt": (0, 50),
    "gh": (0, 50),
    "gleak": (0, 50),
    "erev": (-80, 0),
    "kbm": (-0.035, 0),
}

stats_results = []
group_order = ["iMNTB", "TeNT"]
for col in numeric_cols:
    plt.figure(figsize=(3, 6))

    sns.barplot(
        data=combined_df, x="group", y=col,
        hue="group", palette=custom_palette,
        capsize=0.2, errorbar="se", err_kws={'linewidth': 1.0},
        legend=False, order=group_order
    )

    sns.stripplot(
        data=combined_df, x="group", y=col,
        hue="group", palette=custom_palette,
        dodge=False, alpha=0.6, linewidth=1.0,
        edgecolor="black", size=12, legend=False, order=group_order
    )

    # === Extract values by group ===
    values_imntb = combined_df[combined_df["group"] == "iMNTB"][col].dropna()
    values_tent = combined_df[combined_df["group"] == "TeNT"][col].dropna()

    # === Normality check ===
    # stat_imntb, p_imntb = shapiro(values_imntb)
    # stat_tent, p_tent = shapiro(values_tent)
    # normal = p_imntb > 0.05 and p_tent > 0.05

    # === Select statistical test ===
    # if normal:
    #     stat, pval = ttest_ind(values_imntb, values_tent, equal_var=False)
    #     test_name = "t-test"

    stat, pval = mannwhitneyu(values_imntb, values_tent, alternative="two-sided")
    test_name = "M-W U"

    # === Determine asterisk level ===
    if pval < 0.001:
        p_text = "***"
    elif pval < 0.01:
        p_text = "**"
    elif pval < 0.05:
        p_text = "*"
    else:
        p_text = ""

    # === Add asterisk above bar ===
    if p_text:
        ymax = ylim_dict[col][1] if col in ylim_dict else combined_df[col].max()
        plt.text(0.5, ymax * 0.95, p_text, ha="center", va="top", fontsize=14)

    # === Apply y-axis limits ===
    if col in ylim_dict:
        plt.ylim(ylim_dict[col])
    elif combined_df[col].min() >= 0:
        plt.ylim(bottom=0)

    # === Clean black axes with ticks ===
    ax = plt.gca()
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1)
        ax.spines[spine].set_color("black")
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.tick_params(
        axis='both', which='both',
        bottom=True, left=True,
        top=False, right=False,
        direction='out',
        width=1, length=5,
        color='black'
    )

    stats_results.append({
        "Variable": col,
        "Test": test_name,
        "p-value": pval,
        "Group 1": "iMNTB",
        "Group 2": "TeNT"
        # "Normality iMNTB p": p_imntb,
        # "Normality TeNT p": p_tent,
        # "Used parametric test": normal
    })

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"barplot_{col}.pdf")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📄 Saved: {plot_path}")

# === Save stats after all plots ===
stats_df = pd.DataFrame(stats_results)
stats_path = os.path.join(output_dir, "group_comparison_stats_MW.csv")
stats_df.to_csv(stats_path, index=False)
print(f"📄 Statistical summary saved at: {stats_path}")

# === Compute and save transposed averages separately for iMNTB and TeNT ===
avg_imntb = combined_df[combined_df["group"] == "iMNTB"].mean(numeric_only=True)
avg_tent = combined_df[combined_df["group"] == "TeNT"].mean(numeric_only=True)
med_imntb = combined_df[combined_df["group"] == "iMNTB"].median(numeric_only=True)
med_tent = combined_df[combined_df["group"] == "TeNT"].median(numeric_only=True)
# Transpose for readability
avg_imntb_df = avg_imntb.to_frame(name="avg_iMNTB").T
avg_tent_df = avg_tent.to_frame(name="avg_TeNT").T
med_imntb_df = med_imntb.to_frame(name="med_iMNTB").T
med_tent_df = med_tent.to_frame(name="med_TeNT").T
# Save to CSV
avg_imntb_path = os.path.join(output_dir, "avg_iMNTB_transposed.csv")
avg_tent_path = os.path.join(output_dir, "avg_TeNT_transposed.csv")
med_imntb_path = os.path.join(output_dir, "med_iMNTB_transposed.csv")
med_tent_path = os.path.join(output_dir, "med_TeNT_transposed.csv")

avg_imntb_df.to_csv(avg_imntb_path, index=False)
avg_tent_df.to_csv(avg_tent_path, index=False)
med_imntb_df.to_csv(med_imntb_path, index=False)
med_tent_df.to_csv(med_tent_path, index=False)
print(f"📊 Transposed average iMNTB saved at: {avg_imntb_path}")
print(f"📊 Transposed average TeNT saved at: {avg_tent_path}")
print(f"📊 Transposed median iMNTB saved at: {med_imntb_path}")
print(f"📊 Transposed median TeNT saved at: {med_tent_path}")

