import os
import matplotlib.pyplot as plt
import numpy as np
# from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB
import sys
import datetime
import pandas as pd
h.load_file("stdrun.hoc")
h.dt = 0.02
# === SETTINGS ===
save_figures = True
show_figures = False
filename = ("all_fitted_params_sweep_11_clipped_510ms_02062024_P9_FVB_PunTeTx_Dan_TeNT_120pA_S4C1_20250624_103040.csv").split(".")[0]
filenames = [
    "avg_TeNT_transposed.csv",
    "avg_iMNTB_transposed.csv",
    "median_iMNTB_transposed.csv",
    "median_TeNT_transposed.csv",
    "all_fitted_params_sweep_13_clipped_510ms_02072024_P9_FVB_PunTeTx_Dan_iMNTB_160pA_S3C3_20250624_154105.csv", #ok
    "all_fitted_params_sweep_11_clipped_510ms_02062024_P9_FVB_PunTeTx_Dan_TeNT_120pA_S4C1_20250624_103040.csv", #ok
    "all_fitted_params_sweep_22_clipped_510ms_12232024_P9_FVB_PunTeTx_Dan_TeNT_120pA_S1C1_20250624_103219.csv", #ok
    "all_fitted_params_sweep_8_clipped_510ms_10142022_P9_FVB_PunTeTx_TeNT_60pA_S1C1_20250624_103404.csv", #ok
    "all_fitted_params_sweep_10_clipped_510ms_12172022_P9_FVB_PunTeTx_TeNT_100pA_S2C4_20250624_103550.csv", #ok
    "all_fitted_params_sweep_12_clipped_510ms_03232022_P9_FVB_PunTeTx_TeNT_140pA_S1C2_20250624_103651.csv", #ok
    "all_fitted_params_sweep_17_clipped_510ms_08122022_P9_FVB_PunTeTx_iMNTB_240pA_S1C2_20250624_122608.csv", #ok
    "all_fitted_params_sweep_16_clipped_510ms_12172022_P9_FVB_PunTeTx_iMNTB_220pA_S2C2_20250624_122451.csv", #ok
    "all_fitted_params_sweep_14_clipped_510ms_08122022_P9_FVB_PunTeTx_iMNTB_180pA_S2C1_20250624_122523.csv", #ok
    "all_fitted_params_sweep_16_clipped_510ms_08122022_P9_FVB_PunTeTx_iMNTB_220pA_S1C3_20250624_122402.csv" #ok
]

# === Create Output Folder ===
timestamp = datetime.datetime.now().strftime("%Y%m%d")
output_dir = os.path.join(os.getcwd(),"..", "results","_test_mod_kv4", f"simulation_{filename}")
os.makedirs(output_dir, exist_ok=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
script_dir = os.path.dirname(os.path.abspath(__file__))
param_file_path = os.path.join(script_dir, "..","results","_fit_results","_latest_all_fitted_params", f"{filename}.csv")

if os.path.exists(param_file_path):
    params_df = pd.read_csv(param_file_path)

    # Read expected columns — these must match the CSV header
    gleak = float(params_df.loc[0, "gleak"])
    gklt  = float(params_df.loc[0, "gklt"])
    gh    = float(params_df.loc[0, "gh"])
    erev  = float(params_df.loc[0, "erev"])
    gka   = float(params_df.loc[0, "gka"])
    # Optional: Add these if your `all_fitted_params.csv` has them
    if "gna" in params_df.columns:
        gna = float(params_df.loc[0, "gna"])
    if "gkht" in params_df.columns:
        gkht = float(params_df.loc[0, "gkht"])
    if "cam" in params_df.columns:
        cam = float(params_df.loc[0, "cam"])
    if "kam" in params_df.columns:
        kam = float(params_df.loc[0, "kam"])
    if "cbm" in params_df.columns:
        cbm = float(params_df.loc[0, "cbm"])
    if "kbm" in params_df.columns:
        kbm = float(params_df.loc[0, "kbm"])

    if "cah" in params_df.columns:
        cah = float(params_df.loc[0, "cah"])
    if "kah" in params_df.columns:
        kah = float(params_df.loc[0, "kah"])
    if "cbh" in params_df.columns:
        cbh = float(params_df.loc[0, "cbh"])
    if "kbh" in params_df.columns:
        kbh = float(params_df.loc[0, "kbh"])

    if "can" in params_df.columns:
        can = float(params_df.loc[0, "can"])
    if "kan" in params_df.columns:
        kan = float(params_df.loc[0, "kan"])
    if "cbn" in params_df.columns:
        cbn = float(params_df.loc[0, "cbn"])
    if "kbn" in params_df.columns:
        kbn = float(params_df.loc[0, "kbn"])

    if "cap" in params_df.columns:
        cap = float(params_df.loc[0, "cap"])
    if "kap" in params_df.columns:
        kap = float(params_df.loc[0, "kap"])
    if "cbp" in params_df.columns:
        cbp = float(params_df.loc[0, "cbp"])
    if "kbp" in params_df.columns:
        kbp = float(params_df.loc[0, "kbp"])

    print(f"📥 Loaded best-fit params!")
else:
    raise FileNotFoundError(f"Parameter file not found at {param_file_path}")

script_directory = os.path.dirname(os.path.abspath(__file__))
# Change the working directory to the script's directory
os.chdir(script_directory)
print("Current working directory:", os.getcwd())
#
# gklt = gklt + 2.8
# gna = gna - 1
# gkht = gkht - 70
gka = gka
print("cam",cam)
print("cbm", cbm)
print("kbm", kbm)
print("kam", kam)


# cam = 68.83736049470683
# cbm = 5.31804051571071
# kbm = -0.0340054518969792
# kam = 0.0479514690710309

relaxation = 200
totalcap = 25  # Total membrane capacitance in pF for the cell (input capacitance)
somaarea = (totalcap * 1e-6) / 1   # pf -> uF,assumes 1 uF/cm2; result is in cm2
h.celsius = 35
ek = -106.81
ena = 62.77

cah = 0.000533  #( / ms)
kah = -0.0909   #( / mV)
cbh = 0.787     #( / ms)
kbh = 0.0691    #( / mV)
############################################## stimulus amplitude ######################################################
amps = np.round(np.arange(-0.100, 0.55, 0.020), 3)  # stimulus (first, last, step) in nA
################################### setup the current-clamp stimulus protocol ##########################################
stimdelay: int = 10 + relaxation
stimdur: int = 300
totalrun: int = 510 + relaxation

v_init: int = -75  # if use with custom_init() the value is not considered, but must be close the expected rmp

################################### where to pick the values up the voltages traces to average
t_min = stimdelay + stimdur - 60
t_max = stimdelay + stimdur - 10
average_soma_values = np.array([])  # Array with the average_soma_values in different amplitudes
rmp = None  ######### resting membrane potential measure at steady state when amp = 0 pA

################################ variables to turn on (1) and off (0) blocks of code ###################################
annotation: int = 1  # annotate the variables in the fig 1
insetccfig: int = 1  # cc stim inset in the fig1
apcount: int = 1  # use netcon from NEURON to count APs - it depends currentclamp = 1

plotstimfig: int = 0  # plot the CC stim in fig2
plot_Rin_vs_current = 1  # plot the input resistance vs current graph
plotapcountN: int = 1  # plot the AP counting vs current (NetCon approach)

AP_Rheo: int = 1
AP_phase_plane: int = 1
AP_1st_trace: int = 1
dvdt_plot: int = 1
############################################# MNTB_PN file imported ####################################################
my_cell =  MNTB(0,somaarea,erev,gleak,ena,gna,gh,gka,gklt,gkht,ek,cam,kam,cbm,kbm,cah,kah,cbh,kbh)

##################################### arrays to count APs and detect the rheobase ######################################
ap_counts = []
ap_times = []
trace_data_apc = []

########################################### NEURON approach to detect APs ##############################################
netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
netcon.threshold = 0  # Set the threshold for spike detection

### List to store spike times
spike_times = h.Vector()
netcon.record(spike_times)
num_spikes = []
first_trace_detected = False
first_trace_data = None

################################################## FIGURES 1 AND 2 #####################################################
### fig1 - the simulation
ax1: object
fig1, ax1 = plt.subplots()

### inset with the current stim in fig1
if insetccfig == 1:
    axin = ax1.inset_axes([0.6, 0.1, 0.2, 0.2])  # Create inset of current stimulation in the voltage plot
    ax1.set_xlabel('t (ms)')
    ax1.set_ylabel('v (mV)')
    # axin.set_xlabel('t (ms)')
    axin.set_ylabel('I (nA)')
    axin.grid(False)

### fig2 - the stimulus traces fig2
if plotstimfig == 1:
    ax2: object
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('t (ms)')
    ax2.set_ylabel('I (nA)')

############################################## current clamp simulation ################################################
voltage_dict  = {}
time_vector = None
############################################### CURRENT CLAMP setup ####################################################
stim = h.IClamp(my_cell.soma(0.5))
stim_traces = h.Vector().record(stim._ref_i)
soma_v = h.Vector().record(my_cell.soma(0.5)._ref_v)
t = h.Vector().record(h._ref_t)
for amp in amps:
    mFun.custom_init(v_init)  # default -70mV
    soma_values, stim_values, t_values = mFun.run_simulation(amp, stim, soma_v, t,
                                                             totalrun, stimdelay, stimdur,
                                                             stim_traces)
    soma_values_range, t_values_range, average_soma_values = mFun.avg_ss_values(soma_values, t_values, t_min, t_max,
                                                                                average_soma_values)
    num_spikes,spike_times,ap_counts,ap_times = mFun.count_spikes(num_spikes, stimdelay, stimdur,
                                                                  spike_times,ap_counts, ap_times)

    trim_mask = t_values >= relaxation
    t_plot = t_values[trim_mask] - (stimdelay - 10)
    v_plot = soma_values[trim_mask]
    i_plot = stim_values[trim_mask]

    voltage_dict[f"{amp} nA"] = v_plot
    if time_vector is None:
        time_vector = t_plot
    # if stim_column is None:
    #     stim_column = stim_values

    # === Plot voltage ===
    ax1.plot(t_plot, v_plot, color='red', linewidth=0.5)

    # === Plot inset current ===
    if insetccfig == 1:
        axin.plot(t_plot, i_plot, color='black', linewidth=0.5)

    # === Optional full current plot ===
    if plotstimfig == 1:
        ax2.plot(t_plot, i_plot, color='black', linewidth=0.5)
        ax2.spines['right'].set_color('black')
        ax2.spines['left'].set_color('black')
        ax2.yaxis.label.set_color('black')
        ax2.tick_params(axis='y', colors='black')

# Build DataFrame
df_voltage = pd.DataFrame(voltage_dict, index=time_vector)
df_voltage.index.name = 'Time (s)'


for amp, avg in zip(amps, average_soma_values):
    if 0 == amp:
        rmp = avg

# Calculate the slope for each step in amps relative to the avg_soma_values
slopes = np.array([])
for i in range(1, len(amps)):
    delta_amp = amps[i] - amps[i - 1]
    delta_soma = average_soma_values[i] - average_soma_values[i - 1]
    slope = np.round((delta_soma / delta_amp) / 1000, 3)
    slopes = np.append(slopes, slope)


def calculate_input_resistance_between_minus20_plus20(amps, voltages):
    """Calculate Input Resistance between -20pA and +20pA injections"""
    idx_minus20 = np.argmin(np.abs(amps + 0.02))  # find index closest to -20pA (-0.02nA)
    idx_plus20 = np.argmin(np.abs(amps - 0.02))  # find index closest to +20pA (+0.02nA)

    # Get voltages
    v_minus20 = voltages[idx_minus20]
    v_plus20 = voltages[idx_plus20]

    # Get currents
    i_minus20 = amps[idx_minus20]
    i_plus20 = amps[idx_plus20]

    delta_v = v_plus20 - v_minus20  # mV
    delta_i = i_plus20 - i_minus20  # nA

    # Avoid division by zero
    if np.abs(delta_i) > 1e-9:
        input_resistance = (delta_v / delta_i) / 1000  # GΩ
        return input_resistance
    else:
        print("⚠️ delta_i is too small to calculate input resistance.")
        return None
# Find the slope between the currents you want (default: -0.02 and 0)
#slope_range_index = np.where((np.isclose(amps[:-1], -0.02)) & (np.isclose(amps[1:], 0.0)))[0]
# if len(slope_range_index) > 0:
#     input_resistance = slopes[slope_range_index[0]]
# else:
#     input_resistance = None
input_resistance = calculate_input_resistance_between_minus20_plus20(amps, average_soma_values)
if input_resistance is not None:
    print(f"Input Resistance (-20pA to +20pA): {input_resistance:.3f} GΩ")
else:
    print("Input resistance could not be calculated.")
############################# Arguments: text, xy (point to annotate), xytext (position of the text)
print(f"rmp={rmp}, input_resistance={input_resistance}, gleak={gleak}, gna={gna}, gh={gh}, gklt={gklt}, gkht={gkht}, erev={erev}, ek={ek}, ena={ena}")

if annotation == 1:
    annotation_text = \
    f"""RMP: {rmp: .2f} mV
    Rin: {input_resistance:.3f} GOhms
    gLeak: {gleak:.2f} nS
    gNa: {gna:.2f} nS
    gIH: {gh:.2f} nS
    gKA: {gka:.2f} nS
    gKLT: {gklt:.2f} nS
    gKHT: {gkht:.2f} nS
    ELeak: {erev:.2f} mV
    Ek: {ek:.2f} mV
    ENa: {ena:.2f} mV"""

    ax1.annotate(
        annotation_text,
        xy=(100, -80),  # Point to annotate (x, y)
        xytext=(300, -50),  # Position of the text (x, y)
        # arrowprops=dict(facecolor='black', shrink=0.05),  # Arrow style
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow")  # Add a box around the text
    )

############################### Print the average soma values for each amplitude and the slopes ########################
for amp, avg in zip(amps, average_soma_values):
    print(f"Amplitude: {amp}, Average Soma Value: {avg}")

print("Slopes between consecutive amps and avg_soma_values:")
for i, slope in enumerate(slopes):
    print(f"Slope between amps {amps[i]} and {amps[i + 1]}: {slope} GOhms")

amps_mid_points = (amps[:-1] + amps[1:]) / 2
#####################################################################################################################
if plot_Rin_vs_current == 1:
    # Plot the slopes against the mid-points of amps
    ax3: object
    fig3, ax3 = plt.subplots()
    plt.plot(amps_mid_points, slopes, marker='o', linestyle='-', color='k')
#    mngr.window.setGeometry(1350, 100, 640, 545)
    plt.xlabel('Current (nA)')
    plt.ylabel('Input Resistance (GΩ)')
    plt.title('Input Resistance vs Current Injection')
    plt.grid(True)
#####################################################################################################################
if apcount == 1:
    # Arrays to store AP counts and times
    ap_counts = []
    ap_times = []
    trace_data_apc = []
    # Create a NetCon object to detect spikes
    netcon = h.NetCon(my_cell.soma(0.5)._ref_v, None, sec=my_cell.soma)
    netcon.threshold = -10  # Set the threshold for spike detection

    # List to store spike times
    spike_times = h.Vector()
    netcon.record(spike_times)

    first_trace_detected = False
    first_trace_data = []
    plt.figure()

    for amp in amps:
        mFun.custom_init(v_init)
        soma_values_apc, t_values_apc = mFun.run_simulation(amp, stim, soma_v, t, totalrun, stimdelay, stimdur)

        # Count spikes detected by NetCon
        num_spikes = sum(stimdelay <= time <= stimdelay + stimdur for time in spike_times)
        ap_counts.append(num_spikes)
        ap_times.append(list(spike_times))

        # Extract recorded data
        soma_values_apc = np.array(soma_v.to_python())
        t_values_apc = np.array(t.to_python())


        trim_mask_apc = t_values_apc >= relaxation
        t_plot_apc = t_values_apc[trim_mask_apc] - (stimdelay - 10)
        v_plot_apc = soma_values_apc[trim_mask_apc]
        trace_data_apc.append((t_plot_apc, v_plot_apc, amp, num_spikes))

        # Store the first trace with an AP
        if not first_trace_detected and num_spikes > 0:
            first_trace_data = (t_plot_apc, v_plot_apc, amp)
            first_trace_detected = True

            if AP_Rheo == 1:
                ap_data = mFun.analyze_AP(t_plot_apc, v_plot_apc)

            if AP_phase_plane == 1:
                # === Phase Plane Plot ===
                v_rheo = v_plot_apc
                t_rheo = t_plot_apc
                dv_dt = np.gradient(v_rheo, t_rheo)

                fig_pp, ax_pp = plt.subplots()
                ax_pp.plot(v_rheo, dv_dt, color="darkgreen", linewidth=1)
                ax_pp.set_title(f"Phase Plane: Rheobase AP ({amp * 1000:.0f} pA)")
                ax_pp.set_xlabel("Membrane Voltage (mV)")
                ax_pp.set_ylabel("dV/dt (mV/ms)")
                ax_pp.grid(True)

                # === Optional markers ===
                if ap_data:
                    ax_pp.scatter(ap_data["threshold"], np.interp(ap_data["threshold"], v_rheo, dv_dt),
                                  color='blue', label="Threshold", zorder=5)
                    ax_pp.scatter(ap_data["peak"], np.interp(ap_data["peak"], v_rheo, dv_dt),
                                  color='red', label="Peak", zorder=5)
                    ax_pp.scatter(ap_data["AHP"], np.interp(ap_data["AHP"], v_rheo, dv_dt),
                                  color='purple', label="AHP", zorder=5)

                ax_pp.legend()

                if save_figures:
                    fig_pp.savefig(os.path.join(output_dir, "phase_plane_rheobase.png"), dpi=300, bbox_inches='tight')
                    print("💾 Saved: phase_plane_rheobase.png")
                if show_figures:
                    continue
                plt.close(fig_pp)

        # Clear spike times for the next run
        spike_times.clear()

# Plot all red traces first
if AP_Rheo == 1:
    for t_plot_apc, v_plot_apc, amp, num_spikes in trace_data_apc:
        if num_spikes == 0 or (first_trace_data is not None and (t_plot_apc == first_trace_data[0]).all()):
            plt.plot(t_plot_apc, v_plot_apc, color='red', linewidth=0.5)
            #mngr.window.setGeometry(50, 700, 640, 545)
################################## Plot the first trace with the first AP in black #####################################
    if first_trace_data is not None:
        t_plot_apc, v_plot_apc, amp = first_trace_data
        plt.plot(t_plot_apc, v_plot_apc, color='black', label=f'Rheobase {amp * 1000} pA', linewidth=0.5)
        #mngr.window.setGeometry(700, 700, 640, 545)
    # Display AP counts and times for each amplitude
    for amp, count in zip(amps, ap_counts):
        print(f"Amplitude: {amp} nA, AP Count NetCon: {count}")

    # Plot the results
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')

    plt.legend()

    # Convert the AP counts and times to numpy arrays for further analysis if needed
    ap_counts_array = np.array(ap_counts)
    ap_times_array = np.array(ap_times, dtype=object)

##################################### Plot number of APs vs. stimulus amplitude ########################################
    if plotapcountN == 1:
        ax4: object
        fig4, ax4 = plt.subplots()
        plt.plot(amps * 1000, ap_counts, marker='o', linestyle='-', color='b')
        #mngr.window.setGeometry(1350, 700, 640, 545)  # Sixth window (bottom-right)
        # Set the x-axis tick distance

        ax4.xaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))
        ticks = np.arange(min(amps * 1000), max(amps * 1000) + 20, 20)
        ax4.set_xticks(ticks)

        # Ensure grid lines follow the ticks
        ax4.grid(which='both', axis='both', linestyle='-', linewidth=0.5, c='lightgrey')

        plt.xlabel('Stimulus Amplitude (pA)')
        plt.ylabel('Number of APs')
        plt.title('Number of APs vs. Stimulus Amplitude')
        plt.grid(True)

################################################### AP Analysis ########################################################
if AP_1st_trace == 1:
    if ap_data:
        print("First AP Analysis:")
        for key, value in ap_data.items():
            print(f"{key}: {value:.2f}")

            # Plot the trace with AP features
            fig_ap, ax_ap = plt.subplots()
            ax_ap.plot(t_plot_apc, v_plot_apc, label="Voltage Trace", color='black')
            ax_ap.scatter(ap_data["spike time"], ap_data["peak"], color='red', label="Peak", zorder=3)
            ax_ap.scatter(
                t_plot_apc[np.where(v_plot_apc == ap_data["threshold"])[0][0]],
                ap_data["threshold"],
                color='blue', label="Threshold", zorder=3
            )
            ax_ap.axhline(ap_data["AHP"], color='purple', linestyle="--", label="AHP", alpha=0.7)

            ax_ap.set_xlabel("Time (ms)")
            ax_ap.set_ylabel("Voltage (mV)")
            ax_ap.set_title("Action Potential Analysis")
            ax_ap.legend()

            fig_ap.savefig(os.path.join(output_dir, "AP_features.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_ap)

    else:
        print("No AP detected in this trace.")

# df_voltage.to_csv(os.path.join(output_dir, "voltage_traces.csv"))
if save_figures:
    print(f"\n💾 Saving figures to: {output_dir}\n")

    # Try to save each figure if it exists
    try:
        fig1.savefig(os.path.join(output_dir, "trace_voltage_all_currents.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print("⚠️ fig1 not saved:", e)

    try:
        fig3.savefig(os.path.join(output_dir, "input_resistance_vs_current.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print("⚠️ fig3 not saved:", e)

    try:
        fig4.savefig(os.path.join(output_dir, "AP_count_vs_current.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print("⚠️ fig4 not saved:", e)

    try:
        plt.figure(3)  # This is the AP analysis figure
        plt.savefig(os.path.join(output_dir, "AP_features_rheobase.png"), dpi=300, bbox_inches='tight')
    except Exception as e:
        print("⚠️ AP feature plot not saved:", e)

    import pandas as pd

    # Save steady-state voltages and input resistance
    df = pd.DataFrame({
        "Stimulus_nA": amps,
        "SteadyStateVoltage_mV": average_soma_values,
        "AP_count": ap_counts
    })
    df.to_csv(os.path.join(output_dir, "summary_data.csv"), index=False)

    # Save slopes
    pd.DataFrame({
        "Mid_Current_nA": amps_mid_points,
        "InputResistance_GOhm": slopes
    }).to_csv(os.path.join(output_dir, "input_resistance_curve.csv"), index=False)

    if AP_Rheo and ap_data:
        pd.DataFrame([ap_data]).to_csv(os.path.join(output_dir, "ap_features.csv"), index=False)

    if first_trace_data is not None:
        t_rheo, v_rheo, amp_rheo = first_trace_data
        df_rheo = pd.DataFrame({"Time_ms": t_rheo, "Voltage_mV": v_rheo})
        df_rheo.to_csv(os.path.join(output_dir, "rheobase_trace.csv"), index=False)
    if dvdt_plot:
        df_dvdt = pd.DataFrame({"time_ms": t_rheo, "dvdt": dv_dt})
        df_dvdt.to_csv(os.path.join(output_dir, "dvdt_trace.csv"), index=False)
        plt.figure()
        plt.plot(df_dvdt["time_ms"], df_dvdt["dvdt"])
        plt.xlabel("Time (ms)")
        plt.ylabel("dV/dt (mV/ms)")
        plt.title("dV/dt over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dvdt_trace.png"), dpi=300)
        plt.show()
# === Show Figures in macOS-safe Non-blocking Mode ===
with open(os.path.join(output_dir, "simulation_meta.txt"), "w") as f:
    f.write(f"Experiement: {filename}\n")
    f.write(f"Stimulus Range: {amps[0]} to {amps[-1]} nA\n")
    f.write(f"Stim Duration: {stimdur} ms\n")
    f.write(f"Stim Delay: {stimdelay} ms\n")
    f.write(f"Initial Vm: {v_init} mV\n")
    f.write(f"gLeak: {gleak:.2f} nS\n")
    f.write(f"gNa: {gna:.2f} nS\n")
    f.write(f"gKHT: {gkht:.2f} nS\n")
    f.write(f"gKLT: {gklt:.2f} nS\n")
    f.write(f"gIH: {gh:.2f} nS\n")
    f.write(f"gKA: {gka:.2f} nS\n")
    f.write(f"ELeak: {erev:.2f} mV\n")
    f.write(f"EK: {ek:.2f} mV\n")
    f.write(f"ENa: {ena:.2f} mV\n")
    f.write(f"cam: {cam:.2f} mV\n")
    f.write(f"kam: {kam:.2f} mV\n")
    f.write(f"cbm: {cbm:.2f} mV\n")
    f.write(f"kbm: {kbm:.2f} mV\n")
    # f.write(f"cah: {cah:.2f} mV\n")
    # f.write(f"kah: {kah:.2f} mV\n")
    # f.write(f"cbh: {cbh:.2f} mV\n")
    # f.write(f"kbh: {kbh:.2f} mV\n")
    # f.write(f"can: {can:.2f} mV\n")
    # f.write(f"kan: {kan:.2f} mV\n")
    # f.write(f"cbn: {cbn:.2f} mV\n")
    # f.write(f"kbn: {kbn} .2f mV\n")
    # f.write(f"cap: {cap:.2f} mV\n")
    # f.write(f"kap: {kap:.2f} mV\n")
    # f.write(f"cbp: {cbp:.2f} mV\n")
    # f.write(f"kbp: {kbp:.2f} mV\n")

df_voltage.to_csv(os.path.join(output_dir, "voltage_traces.csv"))

if show_figures:
    plt.show()
    plt.pause(0.001)  # Allow the GUI to update
    sys.stdout.flush()
    input("🔍 Simulation done. Press Enter to close all figures and finish.\n")
    plt.close('all')


