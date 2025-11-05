"""
fit_latency.py

Purpose:
--------
Fit and optimize ion channel conductances and kinetics in a single-compartment
NEURON model to reproduce the experimentally measured *latency to threshold*
for action potential generation, based on current-clamp recordings.

Key Features:
-------------
- Parses metadata from filenames (e.g., postnatal age, phenotype, stimulation amplitude)
- Loads experimental latency-to-threshold and latency-to-peak values from CSV files
- Simulates membrane responses in NEURON for a series of current injections
- Detects latency to threshold using dV/dt threshold crossing
- Fits the model by minimizing the explained sum of squares (ESS) between experimental
  and simulated latency values
- Outputs best-fit parameters, model prediction plots, and fit quality metrics (RMSE, R²)

Inputs:
-------
- Experimental latency data file (CSV) with columns:
    * 'Stimulus (pA)'
    * 'Latency to Threshold (ms)'
    * 'Latency to Peak (ms)'
- Parameter file from previous passive/AP fitting, specific to phenotype

Outputs:
--------
- Best-fit conductance and kinetic parameters
- Plots comparing experimental and simulated latency
- Simulated voltage traces across current steps
- Fit quality metrics (RMSE, R²)

Usage:
------
Run the script by calling:
    fit_latency("example_latency_data.csv", "fitted_param_file.csv")

Dependencies:
-------------
- NEURON
- NumPy, SciPy, pandas, matplotlib
- Custom modules:
    * MNTB_PN_fit.py (defines MNTB model class)
    * MNTB_PN_myFunctions.py (simulation wrapper and helpers)

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # TrueType
from scipy.optimize import minimize, differential_evolution
from scipy.signal import find_peaks
from sklearn.metrics import mean_squared_error, r2_score
from collections import namedtuple
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB

import time
import datetime
import json
import sys

# --- Named tuple to return latency parameters
refinedParams = namedtuple("refinedParams", ["gleak", "gklt", "gh", "gkht", "gna", "gka"])
def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)  # Convert nS to mho/cm²


def fit_latency(filename,param_file):
    start_time = time.time()
    # --- Load experimental data
    file_base = os.path.splitext(os.path.basename(filename))[0]
    # === Extract age from filename
    age_str = "P9"
    for part in file_base.split("_"):
        if part.startswith("P") and part[1:].isdigit():
            age_str = part
            break
    try:
        age = int(age_str[1:])
    except:
        age = 0

    # === Extract phenotype (group)
    if "TeNT" in file_base:
        phenotype = "TeNT"
    elif "iMNTB" in file_base:
        phenotype = "iMNTB"
    else:
        phenotype = "WT"
    print(f"📌 Detected age: {age_str} (P{age}), Phenotype: {phenotype}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data","fit_passive", "fit_passive",filename)
    experimental_data = pd.read_csv(data_path)
    param_file = os.path.join(script_dir, "..", "results", "_fit_results", "_latest_all_fitted_params", f"{phenotype}",
                              param_file)
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Passive parameters not found at: {param_file}")
    print(f"Latency file found at: {filename}")
    print(f"All fitted parameters found at: {param_file}")
    param_df = pd.read_csv(param_file)
    # Extract the first row
    gleak = param_df.loc[0, "gleak"]
    gklt = param_df.loc[0, "gklt"]
    gh = param_df.loc[0, "gh"]
    erev = param_df.loc[0, "erev"]
    gkht = param_df.loc[0, "gkht"]
    gna = param_df.loc[0, "gna"]
    gka = param_df.loc[0, "gka"]
    cam = param_df.loc[0, "cam"]
    kam = param_df.loc[0, "kam"]
    cbm = param_df.loc[0, "cbm"]
    kbm = param_df.loc[0, "kbm"]

    print(f"Loaded parameters: gleak={gleak}, gklt={gklt}, gh={gh}, erev={erev}, gkht={gkht}, gna={gna}, gka={gka}, cam={cam}, kam={kam}, cbm={cbm}, kbm={kbm}")

    lat_thres_col = "Latency to Threshold (ms)"
    lat_peak_col = "Latency to Peak (ms)"
    stim_col = experimental_data["Stimulus (pA)"].values * 1e-3  # pA to nA
    non_null_latency = experimental_data[experimental_data[lat_thres_col].notna()]
    if not non_null_latency.empty:
        rheobase = non_null_latency.iloc[0]["Stimulus (pA)"] * 1e-3
    lat_values = experimental_data[experimental_data[lat_thres_col].notna()].iloc[2:] #starting 40pA more than Rheobase
    fit_currents = lat_values["Stimulus (pA)"].values * 1e-3
    fit_lat_thresh = lat_values[lat_thres_col].values  # ms
    fit_lat_peak = lat_values[lat_peak_col].values #ms

    # --- NEURON setup
    h.load_file("stdrun.hoc")
    h.celsius = 35
    h.dt = 0.02
    v_init = -70

    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm²
    ek = -106.81
    ena = 62.77

    # ################# sodium kinetics
    # cam = 76.4 #76.4
    # kam = .037
    # cbm = 6.930852 #6.930852
    # kbm = -.043

    cah = 0.000533  #( / ms)
    kah = -0.0909   #( / mV)
    cbh = 0.787     #( / ms)
    kbh = 0.0691    #( / mV)

    relaxation = 200

    def run_simulation(p: refinedParams,stim_amp, stim_dur=300, stim_delay=10):
        """
        Run simulation with 200 ms internal relaxation before stimulus.
        Returns the full 710 ms trace, with real stimulus at 210 ms.
        """
        v_init = -75
        totalcap = 25  # pF
        somaarea = (totalcap * 1e-6) / 1  # cm² assuming 1 µF/cm²

        param_dict = {
            "gna": p.gna,
            "gkht": p.gkht,
            "gklt": p.gklt,
            "gh": p.gh,
            "gka": p.gka,
            "gleak": p.gleak,
            "cam": cam,
            "kam": kam,
            "cbm": cbm,
            "kbm": kbm,
            "cah": cah,
            "kah": kah,
            "cbh": cbh,
            "kbh": kbh,
            "erev": erev,
            "ena": ena,
            "ek": ek,
            "somaarea": somaarea
        }

        # Simulate with 200 ms relaxation offset
        t, v = mFun.run_unified_simulation(
            MNTB_class=MNTB,
            param_dict=param_dict,
            stim_amp=stim_amp,
            stim_delay=stim_delay + relaxation,  # ms extra relaxation
            stim_dur=stim_dur,
            v_init=v_init,
            total_duration=510 + relaxation,  # full  ms sim
            return_stim=False
        )

        return t, v

    def extract_latency(t, v, threshold_dvdt=35, stim_start=210):
        dvdt = np.gradient(v, t)
        stim_idx = np.where(t >= stim_start)[0][0]

        # === Threshold latency (dV/dt > threshold)
        above_thresh = np.where(dvdt[stim_idx:] > threshold_dvdt)[0]
        lat_thresh = t[stim_idx + above_thresh[0]] - stim_start if len(above_thresh) > 0 else np.nan

        # === Peak latency using find_peaks (first AP)
        v_seg = v[stim_idx:]
        peaks, _ = find_peaks(v_seg, height=0, distance=5)  # distance prevents counting noise
        lat_peak = t[stim_idx + peaks[0]] - stim_start if len(peaks) > 0 else np.nan

        return lat_thresh, lat_peak

    def compute_ess(params):
        p = refinedParams(*params)
        sim_lat_thresh = []
        sim_lat_peak = []

        # === Extra penalty if cell does not spike at rheobase ===
        p_rheo = refinedParams(*params)
        t_rheo, v_rheo = run_simulation(p_rheo, stim_amp=rheobase)

        dvdt_rheo = np.gradient(v_rheo, t_rheo)
        stim_start_idx = np.where(t_rheo >= 210.5)[0][0]
        above_thresh = np.where(dvdt_rheo[stim_start_idx:] > 35)[0]

        penalty = 0
        if len(above_thresh) == 0:
            penalty += 1000  # No spike at rheobase → heavy penalty
        else:
            latency = t_rheo[stim_start_idx + above_thresh[0]] - 210
            if latency < 0.8:
                penalty += 100 * (0.8 - latency)

        # === Simulate latencies for all fit currents ===
        for I in fit_currents:
            t, v = run_simulation(p, stim_amp=I)
            lat_thresh, lat_peak = extract_latency(t, v)
            sim_lat_thresh.append(lat_thresh)
            sim_lat_peak.append(lat_peak)

        sim_lat_thresh = np.array(sim_lat_thresh)
        sim_lat_peak = np.array(sim_lat_peak)

        valid_thresh = ~np.isnan(sim_lat_thresh) & ~np.isnan(fit_lat_thresh)
        valid_peak = ~np.isnan(sim_lat_peak) & ~np.isnan(fit_lat_peak)

        if np.any(np.isnan(sim_lat_thresh)):
            return 1e6  # Fail-safe: spike detection failed

        ess_thresh = np.sum((fit_lat_thresh[valid_thresh] - sim_lat_thresh[valid_thresh]) ** 2)
        ess_peak = np.sum((fit_lat_peak[valid_peak] - sim_lat_peak[valid_peak]) ** 2)

        return ess_thresh + ess_peak + penalty

    # --- Initial guess and bounds
    print(f"These are the conductance values:"
                 
          f"\n gleak {gleak}"
          f"\n gklt {gklt}"
          f"\n gh {gh}"
          f"\n erev {erev}"
          f"\n gkht {gkht}"
          f"\n gna {gna}"
          f"\n gka {gka}"
          f"\n cam {cam}"
          f"\n kam {kam}"
          f"\n cbm {cbm}"
          f"\n kbm {kbm}")


       # --- Initial guess
    initial = [gleak, gklt, gh, gkht, gna,gka]

    lbgNa = 0.9
    hbgNa = 1.1

    lbKht = 0.8
    hbKht = 1.2

    lbKlt = 0.8
    hbKlt = 1.2

    lbih = 0.9
    hbih = 1.1

    lbleak = 0.9
    hbleak = 1.1

    lbka = 0.5
    hbka = 1.5

    bounds = [
        (gleak * lbleak, gleak * hbleak),  # gleak
        (gklt * lbKlt, gklt * hbKlt),  # gklt
        (gh * lbih, gh * hbih),  # gh
        (gkht * lbKht, gkht * hbKht),  # gkht
        (gna * lbgNa, gna * hbgNa),  # gna
        (gka * lbka, gka * hbka) # gka
    ]
    print("🔍 Running latency optimization...")

    print("🌍 Running global optimization with differential evolution...")

    result_global = differential_evolution(
        compute_ess,
        bounds=bounds,
        strategy='best1bin',
        popsize=20,
        mutation=(0.5, 1),
        recombination=0.7,
        tol=1e-3,
        polish=False,  # We'll polish later
        disp=True
    )

    print(f"🌎 Global optimization complete. ESS = {result_global.fun:.3f}")

    # === Step 2: Refine with L-BFGS-B from global solution
    print("🧪 Refining with local optimization...")

    result_refined = minimize(
        compute_ess,
        x0=result_global.x,
        bounds=bounds,
        method='L-BFGS-B',
        options={
            'disp': True,
            'maxiter': 10000,
            'ftol': 1e-6
        }
    )

    best_params = result_refined.x
    final_cost = result_refined.fun

    print(f"\n✅ Optimization complete. Final ESS = {final_cost:.3f}")
    print("Best-fit params:")
    for name, val in zip(refinedParams._fields, best_params):
        print(f"  {name}: {val:.6f}")



    print(f"\n✅ Optimization complete. Final ESS = {final_cost:.3f}")
    print("Best-fit params:")
    for name, val in zip(refinedParams._fields, best_params):
        print(f"  {name}: {val:.6f}")
    print(f"Sodium Kinetics:"
    f"\n cam {cam}"
    f"\n kam {kam}"
    f"\n cbm {cbm}"
    f"\n kbm {kbm}")

    # === Recompute simulated latencies using best-fit parameters ===
    sim_lat_thresh = []
    for current in fit_currents:
        p = refinedParams(*best_params)
        t, v = run_simulation(p, stim_amp=current)

        dvdt = np.gradient(v, t)
        stim_start_idx = np.where(t >= 210.5)[0][0]
        time_segment = t[stim_start_idx:]
        dvdt_segment = dvdt[stim_start_idx:]

        above_thresh = np.where(dvdt_segment > 35)[0]
        if len(above_thresh) > 0:
            latency = time_segment[above_thresh[0]] - 210
        else:
            latency = np.nan

        sim_lat_thresh.append(latency)

    # === Convert currents back to pA for readability ===
    currents_pA = fit_currents * 1e3

    # === Plot ===
    plt.figure(figsize=(6, 4))
    plt.plot(currents_pA, fit_lat_thresh, 'o-', label='Experimental', linewidth=2)
    plt.plot(currents_pA, sim_lat_thresh, 's--', label='Simulated (fit)', linewidth=2)
    plt.xlabel("Stimulus (pA)")
    plt.ylabel("Latency to Threshold (ms)")
    plt.title("Latency Fit: Experimental vs. Simulated")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Plot simulated voltage traces for a few currents ===

    plt.figure(figsize=(8, 6))

    for current in stim_col:
        p = refinedParams(*best_params)
        t, v = run_simulation(p, stim_amp=current)
        plt.plot(t, v, label=f"{int(current * 1e3)} pA")

    plt.axvline(210, color='gray', linestyle='--', label='Stim Start')
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane Potential (mV)")
    plt.title("Simulated Voltage Traces")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Apply best-fit params

    # --- Fit-specific RMSE and R²
    rmse_fit = np.sqrt(mean_squared_error(fit_lat_thresh, sim_lat_thresh))
    r2_fit = r2_score(fit_lat_thresh, sim_lat_thresh)

    residuals = fit_lat_thresh -sim_lat_thresh


    # --- Save outputs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "..", "results","fit_latency", f"fit_latency_{file_base}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)


# === Optional CLI ===
if __name__ == "__main__":
    import argparse

    # === DEBUG MODE SIMULATION ===
    if sys.gettrace():  # This returns True when running in debugger
        sys.argv = [
            sys.argv[0],  # the script name
            "--data", "/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/data/latency_results/latency_data_02062024_P9_FVB_PunTeTx_TeNT_S4C1.csv",
            "--param_file","/Users/nikollas/Library/CloudStorage/OneDrive-UniversityofSouthFlorida/MNTB_neuron/mod/fit_final/results/_fit_results/_latest_all_fitted_params/all_fitted_params_sweep_11_clipped_510ms_02062024_P9_FVB_PunTeTx_Dan_TeNT_120pA_S4C1_20250624_103040.csv"# your arguments
        ]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to experimental CSV with columns 'Current', 'SteadyStateVoltage'")
    parser.add_argument("--param_file", required=True, help="Path to parameter file")
    args = parser.parse_args()
    fit_latency(args.data,args.param_file)
