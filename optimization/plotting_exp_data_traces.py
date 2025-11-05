import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42   # TrueType
rcParams['ps.fonttype'] = 42    # For EPS too, if needed

filename = "all_sweeps_12172022_P9_FVB_PunTeTx_iMNTB_S2C2.csv".split(".")[0]
csv_path = "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/exported_sweeps/all_sweeps_12172022_P9_FVB_PunTeTx_iMNTB_S2C2.csv"
script_dir = os.path.dirname(os.path.abspath(__file__))
sim_dirs = '/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/exported_sweeps/'

def search_file():
    if sim_dirs:
        latest_folder = max(sim_dirs)
        # voltage_traces = os.path.join(sim_path, latest_folder, "voltage_traces.csv")
        voltage_traces = os.path.join(csv_path)
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
        amps = np.round(np.arange(-0.1, -0.1 + 0.02 * n_sweeps, 0.02), 3)
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
    time_limit = 200  # ms
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
    # col_100pA = df_voltage.columns[11]
    col_200pA = df_voltage.columns[16] #16 for 20 pA steps, 31 for 10pA steps
    if rheobase_col and rheobase_col not in sweep_cols:
        sweep_cols.append(rheobase_col)
    if last_col not in sweep_cols:
        sweep_cols.append(last_col)

    # Step 4: Sort by stimulus order (based on column names)
    sweep_cols_sorted = sorted(sweep_cols, key=lambda x: float(x.split()[0]))

    # Step 5: Plot only up to and including rheobase
    for i, col in enumerate(sweep_cols_sorted):
        trace = df_voltage[col]
        if "iMNTB" in filename:
            color = 'black' if col == rheobase_col else colors[i % 2]
        else:
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
    x_scale = 50  # 50 ms
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
        if script_dir:
            # base_filename = os.path.join(sim_path, sim_dirs, f"{(filename).split('.')[0]}_voltage_traces")
            base_path = os.path.join(script_dir, '..', 'figures')
            new_folder = "exp_data_traces"
            full_path = os.path.join(base_path, new_folder)
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                print(f"Directory created at: {full_path}")
            else:
                print(f"Directory already exists at: {full_path}")
            full_filename = os.path.join(full_path, f"{filename}")
            plt.savefig(f"{full_filename}.png", dpi=dpi, bbox_inches='tight')
            plt.savefig(f"{full_filename}.pdf", dpi=dpi, bbox_inches='tight')
            print(f"✅ Figures saved to:\n{full_filename}.png\n{full_filename}.pdf")
        else:
            print("❌ No matching simulation directory found. Plot not saved.")
    plt.show()

df_voltage = search_file()
plot_voltage_traces(df_voltage,save_fig=True)


