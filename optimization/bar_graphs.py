import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import os
rcParams['pdf.fonttype'] = 42   # TrueType
rcParams['ps.fonttype'] = 42    # For EPS too, if needed
# P4_tentx_exp = {"amp": 40.36, "rheobase": 20}
# P4_imntb_exp = {"amp": 47.15, "rheobase": 110}
# P9_tentx_exp = {"amp": 56.28, "rheobase": 110}
# P9_imntb_exp = {"amp": 43.5,  "rheobase": 210}

P4_tentx_sim = {"amp": 41.84,  "rheobase": 30,  "halfwidth": 0.22,  "threshold": -37.46}
P4_imntb_sim = {"amp": 35.08,  "rheobase": 110, "halfwidth": 0.19,  "threshold": -36.21}
P9_tentx_sim = {"amp": 55.54,  "rheobase": 120, "halfwidth": 0.09,  "threshold": -43.30}
P9_imntb_sim = {"amp": 42.17,  "rheobase": 210, "halfwidth": 0.09,  "threshold": -40.28}


# Group all data by label
labels = ["P4 TeNTx", "P4 iMNTB", "P9 TeNTx", "P9 iMNTB"]

# exp_data = [P4_tentx_exp, P4_imntb_exp, P9_tentx_exp, P9_imntb_exp]
sim_data = [P4_tentx_sim, P4_imntb_sim, P9_tentx_sim, P9_imntb_sim]

features = ["amp","rheobase", "halfwidth", "threshold"]
ylabels = ["Amplitude (mV)", "Rheobase (mV)", "Halfwidth (ms)", "Threshold (mV)"]
bar_colors = ["red", "black", "pink", "gray"]

# Output dir
output_dir = "figures/feature_barplots"
os.makedirs(output_dir, exist_ok=True)
# Convert list of dicts to DataFrame for consistency with conductance plotting
df = pd.DataFrame(sim_data, index=labels)
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# axes = axes.flatten()

for i, feature in enumerate(features):
    values = df[feature].values

    plt.figure(figsize=(3.5, 3.5))
    bars = plt.bar(np.arange(len(labels)), values, color=bar_colors, edgecolor='black')

    # plt.ylabel(ylabels[i])
    plt.title(feature)
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.tick_params(axis='x', labelbottom=False)  # Hide label text for Illustrator

    filename = os.path.join(output_dir, f"{feature}_barplot.pdf")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.1, transparent=True)
    print(f"✅ Saved and showing: {filename}")
    plt.show()
