import pandas as pd
import matplotlib.pyplot as plt
import os
from os.path import split
import numpy as np
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42   # TrueType
rcParams['ps.fonttype'] = 42    # For EPS too, if needed


filename = "04092024_P4_FVB_PunTeTx_TeNTx_tonic.dat".split(".")[0]
exp = "simulation" #the experiment type
script_dir = os.path.dirname(os.path.abspath(__file__))
sim_path = os.path.join(script_dir, "..", "figures")
sim_dirs = [f for f in os.listdir(sim_path) if f.startswith(f"{exp}_{filename}")]

def search_file():
    if sim_dirs:
        latest_folder = max(sim_dirs)
        voltage_traces = os.path.join(sim_path, latest_folder, "voltage_traces.csv")
        if os.path.exists(voltage_traces):
            df_voltage = pd.read_csv(voltage_traces)
            print(f"Found voltage traces in {voltage_traces}")
            return df_voltage
        else:
            print("file does not exist.")
            return None
    return None


def plot_voltage_traces(df_voltage, title="Voltage Traces", xlim=(0,400),ylim=(-120,40), save_fig=False, dpi=300):
    """
    Plots voltage traces from a DataFrame.

    Parameters:
    - df_voltage: pd.DataFrame with time in the first column, and voltage traces in remaining columns.
    - title: Optional title for the plot.
    """
    if df_voltage.columns[1].startswith("Sweep"):
        n_sweeps = df_voltage.shape[1] - 1
        amps = np.round(np.arange(-0.1, -0.1 + 0.01 * n_sweeps, 0.01), 3)
        amp_labels = [f"{amp} nA" for amp in amps]
        df_voltage.columns = [df_voltage.columns[0]] + amp_labels
        sweep_cols = df_voltage.columns[1:].tolist()

    if df_voltage is None or df_voltage.empty:
        print("DataFrame is empty or None. Nothing to plot.")
        return
    print("df dtypes:")
    print(df_voltage.dtypes)
    # Convert all columns except Time to numeric (in-place, handles strings)
    for col in df_voltage.columns[1:]:
        df_voltage[col] = pd.to_numeric(df_voltage[col], errors='coerce')

    plt.figure(figsize=(10, 10))
    time = df_voltage.iloc[:, 0]

    spike_threshold = -20  # mV
    if "iMNTB" in filename:
        colors = ['black', 'gray']
    else:
        colors = ['red', 'pink']
    # Step 1: Find rheobase trace from all sweeps (only check up to 310 ms)
    rheobase_col = None
    time_limit = 310  # ms
    time_mask = time <= time_limit

    for col in df_voltage.columns[1:]:
        trace = df_voltage[col]
        if (trace[time_mask] > spike_threshold).any():
            rheobase_col = col
            print(f"Rheobase detected in: {col} (before {time_limit} ms)")
            break

    sweep_cols = df_voltage.columns[1:].tolist()
    # Step 3: Include rheobase and last trace if skipped
    last_col = df_voltage.columns[-1]
    col_100pA = df_voltage.columns[20]
    col_200pA = df_voltage.columns[31]
    if rheobase_col and rheobase_col not in sweep_cols:
        sweep_cols.append(rheobase_col)
    if last_col not in sweep_cols:
        sweep_cols.append(last_col)

    # Step 4: Sort by stimulus order (based on column names)
    sweep_cols_sorted = sorted(sweep_cols, key=lambda x: float(x.split()[0]))

    # Step 5: Plot only up to and including rheobase
    for i, col in enumerate(sweep_cols_sorted):
        trace = df_voltage[col]
        color = 'red' if col == rheobase_col else colors[i % 2]
        plt.plot(time, trace, color=color, linewidth=1)

        if col == rheobase_col:
            break  # ✅ stop after rheobase
    if col_200pA != rheobase_col:
        trace = df_voltage[col_200pA]
        if "iMNTB" in filename:
            plt.plot(time, trace, color='gray', linewidth=1, linestyle=':')
        else:
            plt.plot(time, trace, color='pink', linewidth=1, linestyle=':')
    # if col_100pA != rheobase_col:
    #     trace = df_voltage[col_100pA]
    #     plt.plot(time, trace, color='pink', linewidth=1, linestyle='solid')

    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    #plt.title(title)
    if len(df_voltage.columns[1:]) <= 10:  # Avoid messy legend with too many traces
        plt.legend(loc="upper right", fontsize="small", ncol=2)
    x_scale = 50  # 10 ms
    y_scale = 20  # 20 mV

    plt.xlim(xlim)
    plt.ylim(ylim)
    xlim_applied = xlim
    ylim_applied = ylim

    # Position (adjust as needed)
    x_start = xlim_applied[1] - 2.51 * x_scale
    y_start = ylim_applied[0] + 0.03 * (ylim_applied[1] - ylim_applied[0])

    # Draw horizontal (time) scale bar
    plt.hlines(y=y_start, xmin=x_start, xmax=x_start + x_scale, linewidth=2, color='black')
    plt.text(x_start + x_scale / 2, y_start - 0.03 * (ylim[1] - ylim[0]), f"{x_scale} ms",
             ha='center', va='top', fontsize=14)

    # Draw vertical (voltage) scale bar
    plt.vlines(x=x_start, ymin=y_start, ymax=y_start + y_scale, linewidth=2, color='black')
    plt.text(x_start - 0.01 * (xlim[1] - xlim[0]), y_start + y_scale / 2, f"{y_scale} mV",
             ha='right', va='center', fontsize=14,rotation=90)

    plt.grid(False)
    plt.tight_layout()
    plt.axis('off')
    if save_fig:
        # Build the output path

        if sim_dirs:
            latest_dir = max(sim_dirs)
            base_filename = os.path.join(sim_path, latest_dir, f"{filename}_200pA_voltage_traces")
            plt.savefig(f"{base_filename}.png", dpi=dpi, bbox_inches='tight')
            plt.savefig(f"{base_filename}.pdf", dpi=dpi, bbox_inches='tight')
            print(f"✅ Figures saved to:\n{base_filename}.png\n{base_filename}.pdf")
        else:
            print("❌ No matching simulation directory found. Plot not saved.")
    plt.show()


def extract_conductances(file_paths, selected_keys=None):
    """
    Extract specified conductance values from CSV files.

    Args:
        file_paths (list of str): Paths to CSV files.
        selected_keys (list of str): Conductance keys to extract. If None, extract all columns.

    Returns:
        pd.DataFrame: A dataframe with filenames as index and conductance values as columns.
    """
    data = []
    labels = []

    for path in file_paths:
        df = pd.read_csv(path)
        if selected_keys is None:
            values = df.iloc[0].to_dict()
        else:
            values = {k: df.iloc[0][k] for k in selected_keys if k in df.columns}

        parts = os.path.basename(path).split("_")
        label = "_".join(parts[8:13])  # Custom descriptive label
        data.append(values)
        labels.append(label)
    return pd.DataFrame(data, index=labels)

def plot_conductance_lines(df, title="Conductance Comparison"):
    """
    Plots a line graph comparing conductance values across different samples.

    Args:
        df (pd.DataFrame): DataFrame with rows as samples and columns as conductance types.
        title (str): Plot title.
    """
    plt.figure(figsize=(10, 6))

    for index, row in df.iterrows():
        plt.plot(df.columns, row.values, marker='o', label=index)

    plt.xlabel("Conductance Type")
    plt.ylabel("Value (nS)")
    plt.title(title)
    plt.grid(True)
    plt.legend(title="Sample")
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
# Function to plot stacked bar chart
def plot_stacked_conductance_bars(df, title="Stacked Conductance Comparison"):
    ax = df.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')
    ax.set_xlabel("Sample")
    ax.set_ylabel("Conductance Value (nS)")
    ax.set_title(title)
    ax.legend(title="Conductance Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()

def plot_grouped_conductance_bars(df, title="Conductance Values by Cell"):
    """
    Plots grouped bar plots: one group per conductance, one bar per cell.
    """
    df_T = df.T  # Transpose so rows are conductance types

    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2
    x = np.arange(len(df_T.index))  # Number of conductance types

    for i in range(df_T.shape[1]):
        cell_label = df_T.columns[i]
        ax.bar(x + i * bar_width, df_T.iloc[:, i], width=bar_width, label=cell_label)

    ax.set_xticks(x + bar_width * (len(df_T.columns) - 1) / 2)
    ax.set_xticklabels(df_T.index, rotation=45)
    ax.set_ylabel("Conductance (nS)")
    ax.set_xlabel("Conductance Type")
    ax.set_title(title)
    ax.legend(title="Cell")
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

def plot_grouped_conductance_bars_by_group(df, major_keys, minor_keys, title_prefix="Conductance Values"):
    """
    Plots two grouped bar plots: one for major conductances, one for minor conductances.

    Parameters:
        df: DataFrame with conductance values (cells as rows, conductances as columns)
        major_keys: list of conductance types considered major
        minor_keys: list of conductance types considered minor
        title_prefix: base title for both subplots
    """

    def _plot_group(df_group, keys, ax, title, colors):
        df_T = df_group[keys].T  # Transpose to: rows = conductances, columns = cells
        bar_width = 0.2
        x = np.arange(len(df_T.index))  # number of conductance types

        for i in range(df_T.shape[1]):
            ax.bar(x + i * bar_width, df_T.iloc[:, i],
                   width=bar_width, color=colors[i], label=df_T.columns[i])

        ax.set_xticks(x + bar_width * (df_T.shape[1] - 1) / 2)
        ax.set_xticklabels(df_T.index, rotation=45)
        ax.set_ylabel("Conductance (nS)")
        ax.set_title(title)
        ax.grid(False)



    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6), sharey=False)

    fixed_colors = ['red', 'black', 'pink', 'gray'][:len(df)]  # truncate if fewer than 4

    _plot_group(df, major_keys, ax1, f"{title_prefix} — Major", fixed_colors)
    _plot_group(df, minor_keys, ax2, f"{title_prefix} — Minor", fixed_colors)

    ax1.set_xlabel("Major Conductances")
    ax2.set_xlabel("Minor Conductances")

    # Shared legend
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(df.index), title="Cell")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
def show_and_save_each_conductance(df, output_dir="figures/conductance_types", normalize=False):
    os.makedirs(output_dir, exist_ok=True)

    if normalize:
        df = df.div(df['gna'], axis=0)

    color_order = ['red', 'black', 'pink', 'gray']  # up to 4 cells

    major_keys = ['gna', 'gkht', 'gka']
    minor_keys = ['gklt', 'gh', 'gleak']

    for conductance in df.columns:
        plt.figure(figsize=(3.5,3.5))

        df[conductance].plot(kind='bar',
                             color=color_order[:len(df.index)],
                             edgecolor='black')

        plt.ylabel("Normalized Value" if normalize else "Conductance (nS)")
        plt.title(f"{conductance} across cells")
        plt.xticks(ticks=np.arange(len(df.index)))

        # === Set specific Y-axis limits ===
        if not normalize:
            if conductance in major_keys:
                plt.ylim(0, 250)
            elif conductance in minor_keys:
                plt.ylim(0, 20)

        safe_name = conductance.replace("/", "_")
        filename = os.path.join(output_dir, f"{safe_name}_barplot.pdf")

        plt.savefig(filename, format='pdf')
        print(f"✅ Saved and showing: {filename}")
        plt.show()

# Example usage:
conductance_keys = ['gleak', 'gna', 'gklt', 'gkht', 'gh', 'gka']
file_paths = [
"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/optimization/all_fitted_params_sweep_13_clipped_510ms_04092024_P4_FVB_PunTeTx_TeNTx_tonic_20250512_103843_adapted.csv",
"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/optimization/all_fitted_params_sweep_11_clipped_510ms_02012023_P4_FVB_PunTeTx_tonic_iMNTB_20250502_122959_adapted.csv",
"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/optimization/all_fitted_params_sweep_11_clipped_510ms_12172022_P9_FVB_PunTeTx_tonic_TeNTx_20250502_141241_adapted.csv",
"/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/optimization/all_fitted_params_sweep_15_clipped_510ms_12172022_P9_FVB_PunTeTx_phasic_iMNTB_20250502_134232_adapted.csv"
]

df_voltage = search_file()
plot_voltage_traces(df_voltage,save_fig=True)

df_conductances = extract_conductances(file_paths, conductance_keys)
df_conductances_T = df_conductances.T
df_normalized = df_conductances.div(df_conductances["gna"], axis=0)
df_normalized_no_gna = df_normalized.drop(columns='gna')
df_normalized_T = df_normalized.T

df_no_gna = df_normalized_T.drop(index='gna')
df_no_gna_T = df_no_gna.T
#
# plot_conductance_lines(df_conductances)
# plot_conductance_lines(df_normalized)
# plot_conductance_lines(df_no_gna_T)
# plot_stacked_conductance_bars(df_conductances)
# plot_stacked_conductance_bars(df_conductances_T, title="Stacked Conductance (Unnormalized)")
# plot_stacked_conductance_bars(df_normalized)
# plot_stacked_conductance_bars(df_no_gna_T, title="Stacked Conductance (Normalized to gNa = 1)")
# plot_stacked_conductance_bars(df_normalized_T, title="Stacked Conductance (Normalized to gNa = 1)")
# plot_stacked_conductance_bars(df_no_gna)

# # Define reordered and grouped conductance keys again
major_keys = ['gna', 'gkht', 'gka']
minor_keys = ['gklt', 'gh', 'gleak']
df_ordered = df_conductances[major_keys + minor_keys]

# plot_grouped_conductance_bars(df_conductances)
# plot_grouped_conductance_bars_by_group(df_conductances, major_keys, minor_keys)
# plot_grouped_conductance_bars_by_group(df_normalized_no_gna, ['gkht', 'gka'], ['gklt', 'gh', 'gleak'])
show_and_save_each_conductance(df_conductances,normalize=False)
df_ratios = pd.DataFrame({
    'gkht/gna': df_conductances['gkht'] / df_conductances['gna'],
    'gklt/gna': df_conductances['gklt'] / df_conductances['gna']
}, index=df_conductances.index)
show_and_save_each_conductance(df_ratios, normalize=False)
# # Create two side-by-side subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

# # Define color map for consistency
# colors = plt.cm.tab10.colors
# sample_labels = df_ordered.index.tolist()

# # === Left panel: Major conductances ===
# for i, label in enumerate(sample_labels):
#     ax1.plot(major_keys, df_ordered.loc[label, major_keys], marker='o', color=colors[i], label=label)
# # ax1.set_title("Major Conductances")
# ax1.set_ylim(0, 250)
# ax1.set_ylabel("Conductance (nS)")
# ax1.set_xlabel("Major Conductances")
# ax1.set_xticks(major_keys)
#
# # === Right panel: Minor conductances ===
# for i, label in enumerate(sample_labels):
#     ax2.plot(minor_keys, df_ordered.loc[label, minor_keys], marker='o', color=colors[i])
# # ax2.set_title("Minor Conductances")
# ax2.set_ylim(0, 25)
# ax2.set_xlabel("Minor Conductances")
# ax2.set_xticks(minor_keys)

# Shared legend
# fig.legend(sample_labels, loc='upper center', ncol=len(sample_labels), title="Sample", bbox_to_anchor=(0.5, 1.05))


plt.tight_layout()
plt.show()
