import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42   # TrueType
rcParams['ps.fonttype'] = 42    # For EPS too, if needed
# Load NEURON hoc files
h.load_file('stdrun.hoc')
h.celsius = 35

# === Load fitted parameters ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
param_file_path = os.path.join(os.path.dirname(__file__),"..","..","CSV", "avg_iMNTB_transposed.csv")
filename = "gNa_vs_gKLT_gNa_ratio_TeNT_average"
if os.path.exists(param_file_path):
    params_df = pd.read_csv(param_file_path)
    params_row = params_df.loc[0]

    fixed_params = {
        'gid': 0,
        'somaarea': 25e-6,  # cm² (25 pF assuming 1 µF/cm²)
        'erev': params_row["erev"],
        'gleak': params_row["gleak"],
        'gh': params_row["gh"],
        'gklt': params_row["gklt"],
        'gna': params_row["gna"],
        'gkht': params_row["gkht"],  # this will be updated later
        'gka': params_row["gka"],
        'ena': 62.77,
        'ek': -106.81,
        'cam': params_row["cam"],
        'kam': params_row["kam"],
        'cbm': params_row["cbm"],
        'kbm': params_row["kbm"]
    }

    print("📥 Parameters loaded successfully.")
else:
    raise FileNotFoundError(f"Parameter file not found at: {param_file_path}")
gna_fixed = fixed_params['gna']
print(f"gNa fixed: {gna_fixed}")
gklt_fixed = fixed_params['gklt']
print(f"gKLT fixed: {gklt_fixed}")
ratio_fixed = gklt_fixed / gna_fixed if gna_fixed != 0 else 0.0

# === Define ranges ===
gna_values = np.linspace(20, 400, 50)        # Sodium conductance in nS
ratios = np.linspace(0.0, 0.2, 50)            # gNa/gKLT ratios

spike_matrix = np.zeros((len(ratios), len(gna_values)))

# === Simulation parameters ===
stim_start = 10      # ms
stim_end = 310       # ms
stim_amp = 0.200      # nA
threshold = -15       # mV for spike detection

# === Run simulations ===
for i, ratio in enumerate(ratios):
    for j, gna in enumerate(gna_values):
        gklt = gna * ratio

        # Update parameters
        fixed_params['gna'] = gna
        fixed_params['gklt'] = gklt

        neuron = MNTB(**fixed_params)

        # Inject current
        stim = h.IClamp(neuron.soma(0.5))
        stim.delay = stim_start
        stim.dur = stim_end - stim_start
        stim.amp = stim_amp

        # Record voltage and time
        v = h.Vector().record(neuron.soma(0.5)._ref_v)
        t = h.Vector().record(h._ref_t)

        mFun.custom_init(-70)
        h.continuerun(510)

        v_np = np.array(v)
        t_np = np.array(t)

        # Detect spikes
        spike_indices = np.where((v_np[:-1] < threshold) & (v_np[1:] >= threshold))[0]
        spike_times = t_np[spike_indices]
        valid_spikes = np.logical_and(spike_times >= stim_start, spike_times <= stim_end)
        spike_count = np.sum(valid_spikes)

        # Store result
        spike_matrix[i, j] = spike_count

classification_map = np.zeros_like(spike_matrix)

classification_map[spike_matrix == 0] = 0                    # Silent
classification_map[(spike_matrix >= 1) & (spike_matrix <= 3)] = 1  # Phasic
classification_map[spike_matrix >= 4] = 2                    # Tonic
# === Plotting ===
plt.figure(figsize=(10, 8))
plt.imshow(spike_matrix, origin='lower', aspect='auto',
           extent=[gna_values[0], gna_values[-1], ratios[0], ratios[-1]],
           cmap='viridis', vmin=0, vmax=3)
# === Plot red dot on 2D heatmap ===
plt.scatter(gna_fixed, ratio_fixed, color='red', s=80,
            edgecolor='black', linewidth=1.2, label='Fixed Params')
plt.legend(loc='upper right')
plt.colorbar(label='Number of Spikes')
plt.xlabel('gNa (nS)')
plt.ylabel('gNa / gKLT Ratio')
plt.title('Spike Count vs gNa and gKLT/gNa Ratio - TeNT')
plt.grid(False)
plt.tight_layout()
plt.show()

# Create meshgrid for gNa and ratios
GNA, RATIO = np.meshgrid(gna_values, ratios)

# 3D plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Normalize color range for colormap
norm = plt.Normalize(vmin=0, vmax=4)
colors = plt.cm.viridis(norm(spike_matrix))

# Find closest mesh location to original gna and ratio
i_closest = (np.abs(ratios - ratio_fixed)).argmin()
j_closest = (np.abs(gna_values - gna_fixed)).argmin()

colors[i_closest, j_closest] = [1.0, 0.0, 0.0, 1.0]
from matplotlib import colors as mcolors

# Define number of tiles to highlight outward (i.e., radius)
highlight_radius = 3  # you can make this larger if needed

# Define target color gradient: red center → orange → yellow
fade_colors = [
    mcolors.to_rgba('red'),
    mcolors.to_rgba('orange'),
    mcolors.to_rgba('gold')
]

# Paint concentric rings around the center tile
for di in range(-highlight_radius, highlight_radius + 1):
    for dj in range(-highlight_radius, highlight_radius + 1):
        ii = i_closest + di
        jj = j_closest + dj

        if 0 <= ii < colors.shape[0] and 0 <= jj < colors.shape[1]:
            dist = np.sqrt(di**2 + dj**2)
            level = int(dist)  # 0 = red, 1 = orangered, etc.
            spike =spike_matrix[ii, jj]
            print(f"Offset ({di},{dj}) → {spike:.0f} spikes @ distance {dist:.2f}")
            if level < len(fade_colors):
                colors[ii, jj] = fade_colors[level]

# Plot surface with embedded red tile
surf = ax.plot_surface(GNA, RATIO, spike_matrix,
                       facecolors=colors, rstride=1, cstride=1,
                       linewidth=0.2, rasterized=False, edgecolor='black', antialiased=True, alpha=1.0)

# Axis labels and title
ax.set_xlabel('gNa (nS)')
ax.set_ylabel('gKLT / gNa Ratio')
ax.set_zlabel('Spike Count')
ax.set_zlim(0, 120)  # or use 0 to np.max(spike_matrix) if dynamic but bounded
ax.set_title('3D Surface of Spike Count vs gNa and gKLT/gNa Ratio - TeNT')
ax.view_init(elev=30, azim=150,roll=3)

# Optional: vertical arrow for visual anchoring
ax.plot([gna_values[j_closest]] * 2,
        [ratios[i_closest]] * 2,
        [spike_matrix[i_closest, j_closest], spike_matrix[i_closest, j_closest] + 2],
        color='gray', linestyle='--', linewidth=1)

# === Simulate the fixed point directly ===
fixed_sim_params = fixed_params.copy()
fixed_sim_params['gna'] = gna_fixed
fixed_sim_params['gklt'] = gklt_fixed

neuron_fixed = MNTB(**fixed_sim_params)

stim = h.IClamp(neuron_fixed.soma(0.5))
stim.delay = stim_start
stim.dur = stim_end - stim_start
stim.amp = stim_amp

v_fix = h.Vector().record(neuron_fixed.soma(0.5)._ref_v)
t_fix = h.Vector().record(h._ref_t)

mFun.custom_init(-70)
h.continuerun(510)

v_np_fix = np.array(v_fix)
t_np_fix = np.array(t_fix)

spike_indices_fix = np.where((v_np_fix[:-1] < threshold) & (v_np_fix[1:] >= threshold))[0]
spike_times_fix = t_np_fix[spike_indices_fix]
valid_spikes_fix = np.logical_and(spike_times_fix >= stim_start, spike_times_fix <= stim_end)
spike_fixed = np.sum(valid_spikes_fix)
plt.tight_layout()
os.makedirs("../../figures/3D_graphs", exist_ok=True)
plt.savefig(f"../../figures/3D_graphs/{filename}.pdf", format="pdf", bbox_inches='tight')
plt.show()
plt.figure()
plt.plot(t_np_fix, v_np_fix, label="Fixed Params Trace")
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Voltage (mV)")
plt.title("Voltage Trace at Fixed Params - stim 200 pA")
plt.legend()
plt.tight_layout()
plt.show()
