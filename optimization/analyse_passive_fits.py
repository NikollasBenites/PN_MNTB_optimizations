import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, levene, ttest_ind, kruskal, mannwhitneyu
from itertools import combinations
import os
import  re

# === Load both datasets ===
base_dir = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_passive_results"
tent_path = os.path.join(base_dir, "passive_fit_summary_TeNT.csv")
imntb_path = os.path.join(base_dir, "passive_fit_summary_iMNTB.csv")

df_tent = pd.read_csv(tent_path)
df_imntb = pd.read_csv(imntb_path)

# === Add group labels if not already present ===
df_tent["group"] = "TeNT"
df_imntb["group"] = "iMNTB"

# === Combine both into a single DataFrame ===
df = pd.concat([df_tent, df_imntb], ignore_index=True)


# # === Parse age and clean up ===
df["age_num"] = df["age"].str.extract(r"P(\d+)").astype(float)
df = df[df["r2_fit"] > 0.85]  # Optional: filter low-quality fits

# === Create output directory ===
output_dir = os.path.join(base_dir, "stats_passive")

os.makedirs(output_dir, exist_ok=True)

per_cell_dir = os.path.join(output_dir, "per_cell_barplots")
os.makedirs(per_cell_dir, exist_ok=True)

for source_file in df["source_file"].unique():
    df_cell = df[df["source_file"] == source_file]

    group = df_cell["group"].iloc[0] if not df_cell["group"].isnull().all() else "Unknown"
    cell_id = df_cell["cell_id"].iloc[0] if not df_cell["cell_id"].isnull().all() else "?"

    df_melted = df_cell.melt(
        id_vars=["age", "age_num", "group", "cell_id", "source_file"],
        value_vars=["gleak", "gklt", "gh"],
        var_name="conductance",
        value_name="value"
    )

    if df_melted["value"].isnull().all():
        continue

    plt.figure(figsize=(8, 8))
    sns.barplot(
        data=df_melted,
        x="age_num", y="value",
        hue="conductance",
        errorbar="sd",
        palette="pastel"
    )

    # Clean file name
    clean_name = re.sub(r'^passive_summary_experimental_data_', '', source_file)
    clean_name = re.sub(r'\.json.*$', '', clean_name)
    safe_filename = clean_name.replace("/", "_")

    filename = f"{safe_filename}_{group}.png"
    plt.title(f"{safe_filename} ({group}, {cell_id})")
    plt.xlabel("Age (P)")
    plt.ylabel("Conductance (nS)")
    plt.ylim(0, 50)
    plt.legend(title="Conductance")
    plt.tight_layout()
    plt.savefig(os.path.join(per_cell_dir, filename), dpi=300)
    plt.close()



# === Bar plots with individual data points ===
for param in ["gleak", "gklt", "gh"]:
    custom_palette = {
        "TeNT": "#ff9999",  # light red
        "iMNTB": "gray"  # neutral gray
    }
    plt.figure(figsize=(8, 8))

    # Bar plot with error bars
    sns.barplot(
        data=df,
        x="age_num",
        y=param,
        hue="group",
        errorbar="sd",  # ✅ correct usage
        palette=custom_palette,
        dodge=True
    )

    # Overlay individual data points
    sns.stripplot(
        data=df,
        x="age_num",
        y=param,
        hue="group",
        dodge=True,
        palette=custom_palette,
        alpha=0.6,
        marker="o",
        edgecolor="auto",  # ✅ future-safe
        linewidth=0.5
    )

    plt.title(f"{param} by Age and Phenotype")
    plt.ylabel(f"{param} (nS)")
    plt.xlabel("Age (P)")
    plt.ylim(0, 40)

    # Fix duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    n = len(set(df["group"]))
    plt.legend(handles[:n], labels[:n], title="Group", loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{param}_barplot.png"), dpi=300)
    plt.close()

# === Run ANOVA for each parameter ===
if df["age_num"].nunique() > 1 and df["group"].nunique() > 1:
    anova_results = {}

    for param in ["gleak", "gklt", "gh"]:
        subdf = df[["group", "age_num", param]].dropna()
        if subdf["group"].nunique() < 2 or subdf["age_num"].nunique() < 2:
            print(f"⚠️ Skipping ANOVA for {param}: not enough variation.")
            continue

        model = ols(f"{param} ~ C(group) + C(age_num)", data=subdf).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)

        anova_results[param] = anova_table

        print(f"\n📊 ANOVA for {param}")
        print(anova_table)
        anova_table.to_csv(os.path.join(output_dir, f"anova_{param}.csv"))

    print("\n✅ ANOVA completed and saved.")
else:
    print("⚠️ Skipping ANOVA: not enough groups or ages for factorial analysis.")


# === t - test ===
test_results = []

for param in ["gleak", "gklt", "gh"]:
    subdf = df[["group", "age_num", param]].dropna()
    unique_ages = sorted(subdf["age_num"].unique())
    unique_groups = sorted(subdf["group"].unique())

    for age in unique_ages:
        age_df = subdf[subdf["age_num"] == age]
        groups = [age_df[age_df["group"] == g][param].values for g in unique_groups]

        # Skip if missing data or not enough groups
        if any(len(g) < 2 for g in groups):
            continue

        # Test normality (per group)
        normality = [shapiro(g)[1] > 0.05 for g in groups if len(g) >= 3]
        is_normal = all(normality)

        # Test equal variance
        if len(groups) == 2 and all(len(g) >= 3 for g in groups):
            _, p_var = levene(*groups)
            equal_var = p_var > 0.05
        else:
            equal_var = True

        # === Decide test
        if len(groups) == 2:
            if is_normal and equal_var:
                stat, pval = ttest_ind(*groups, equal_var=True)
                test_name = "t-test"
            else:
                stat, pval = mannwhitneyu(*groups, alternative="two-sided")
                test_name = "Mann-Whitney"
        else:
            if is_normal and equal_var:
                model = ols(f"{param} ~ C(group)", data=age_df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                pval = anova_table["PR(>F)"].iloc[0]
                test_name = "ANOVA"
            else:
                stat, pval = kruskal(*groups)
                test_name = "Kruskal-Wallis"

        test_results.append({
            "param": param,
            "age": age,
            "test": test_name,
            "p_value": pval,
            "normal": is_normal,
            "equal_var": equal_var
        })

# === Save summary of statistical tests
test_df = pd.DataFrame(test_results)
test_df.to_csv(os.path.join(output_dir, "param_vs_nonparam_tests.csv"), index=False)

print("\n📊 Parametric vs non-parametric tests saved.")