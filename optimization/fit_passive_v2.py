"""
fit_passive.py

Fit and optimize passive conductances (gKLT, gH, gLeak) and E_leak to experimental
steady-state current-clamp data using explained sum of squares (ESS) minimization.

Output:
- best_fit_params.txt for legacy use
- passive_params_<input_file>.txt (timestamped)
- passive_summary_<input_file>.json
- plots of VI curve and Rin fits
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # TrueType
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error, r2_score
from collections import namedtuple
from neuron import h
import MNTB_PN_myFunctions as mFun
from MNTB_PN_fit import MNTB

import time
import datetime
import json

# --- Named tuple to return passive parameters
PassiveParams = namedtuple("PassiveParams", ["gleak", "gklt", "gh", "erev", "gkht", "gna", "gka"])


def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)  # Convert nS to mho/cm²


def fit_passive(filename):
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
    phenotype = "WT"
    if "TeNT" in file_base:
        phenotype = "TeNT"
    elif "iMNTB" in file_base:
        phenotype = "iMNTB"

    print(f"📌 Detected age: {age_str} (P{age}), Phenotype: {phenotype}")

    # === Extract upper stimulus limit (e.g., "80pA" → include ≤ 0.080 nA)
    stim_cap_nA = None
    for part in file_base.split("_"):
        if "pA" in part:
            try:
                stim_cap_nA = float(part.replace("pA", "")) * 1e-3  # Convert pA to nA
                break
            except:
                pass
    if stim_cap_nA is not None:
        print(f"⚡ Using stimulus cap: ≤ {stim_cap_nA * 1e3:.0f} pA")
    else:
        print("⚠️ No stimulus cap detected in filename.")


    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data","fit_passive", "fit_passive",filename)
    experimental_data = pd.read_csv(data_path)

    vconverter = 1
    all_currents = experimental_data["Stimulus"].values * 1e-3  # pA to nA
    all_steady_state_voltages = experimental_data["SteadyState (nA or mV)"].values * vconverter  # V to mV

    if stim_cap_nA is not None:
        mask = all_currents <= stim_cap_nA
        fit_currents = all_currents[mask]
        fit_voltages = all_steady_state_voltages[mask]
        print(f"✂️ Clipped to {len(fit_currents)} data points (≤ {stim_cap_nA * 1e3:.0f} pA)")
    else:
        fit_currents = all_currents
        fit_voltages = all_steady_state_voltages

    # --- NEURON setup
    h.load_file("stdrun.hoc")
    h.celsius = 35
    h.dt = 0.02
    v_init = -70

    totalcap = 25  # pF
    somaarea = (totalcap * 1e-6) / 1  # cm²

    soma = h.Section(name='soma')
    soma.L = 20
    soma.diam = 15
    soma.Ra = 150
    soma.cm = 1
    soma.insert('leak')
    soma.insert('HT_dth_nmb')
    soma.insert('LT_dth')
    soma.insert('NaCh_nmb')
    soma.insert('IH_nmb')
    soma.insert('ka')
    soma.ek = -106.8
    soma.ena = 62.77

    st = h.IClamp(soma(0.5))
    st.dur = 300
    st.delay = 10

    v_vec = h.Vector()
    t_vec = h.Vector()
    v_vec.record(soma(0.5)._ref_v)
    t_vec.record(h._ref_t)

    def run_simulation(current_injection):
        st.amp = current_injection
        v_vec.resize(0)
        t_vec.resize(0)
        h.v_init = v_init
        mFun.custom_init(v_init)
        h.tstop = st.delay + st.dur
        h.continuerun(h.tstop)
        time_array = np.array(t_vec)
        voltage_array = np.array(v_vec)
        steady_voltage = np.mean(voltage_array[(time_array >= 250) & (time_array <= 300)])
        return steady_voltage

    def compute_ess(params):
        gleak, gklt, gh, erev, gkht, gna, gka = params
        soma.g_leak = nstomho(gleak, somaarea)
        soma.gkltbar_LT_dth = nstomho(gklt, somaarea)
        soma.ghbar_IH_nmb = nstomho(gh, somaarea)
        soma.erev_leak = erev
        soma.gkhtbar_HT_dth_nmb = nstomho(gkht, somaarea)
        soma.gnabar_NaCh_nmb = nstomho(gna, somaarea)
        soma.gkabar_ka = nstomho(gka, somaarea)
        simulated = np.array([run_simulation(i) for i in fit_currents])
        return np.sum((fit_voltages - simulated) ** 2)

    # --- Initial guess and bounds
    gkht, gna, gka = 10, 10, 10
    # --- Initial guess
    initial = [6.0, 10.0, 4.0, -75.0, gkht, gna, gka]

    # === Adaptive bounds based on (age, phenotype)
    if age <= 3:
        gleak_bounds = (0.5, 8)
        gklt_bounds = (1, 20)
        gh_bounds = (0.1, 10)
    elif age <= 6:
        gleak_bounds = (1, 15)
        gklt_bounds = (5, 30)
        gh_bounds = (0.5, 15)
    else:
        gleak_bounds = (2, 20)
        gklt_bounds = (10, 60)
        gh_bounds = (1, 30)

    # === Apply treatment-specific restrictions
    if phenotype == "TeNT":
        gklt_bounds = (1, min(gklt_bounds[1], 30))  # assume reduced KLT
        gh_bounds = (gh_bounds[0], min(gh_bounds[1], 30))  # clamp HCN upper limit
    elif phenotype == "iMNTB":
        gleak_bounds = (gleak_bounds[0], min(gleak_bounds[1], 12))  # restrict leak range
        gh_bounds = (0.5, 20)  # maybe lower HCN across ages in TeNT group

    # Final optimizer bounds
    bounds = [
        gleak_bounds,
        gklt_bounds,
        gh_bounds,
        (-85, -70),  # E_leak
        (gkht * 0.5, gkht * 2),  # KHT (fixed)
        (gna * 0.5, gna * 2),  # Na (fixed)
        (gka * 0.5, gka * 2)  # KA (fixed)
    ]

    print("🔧 Final bounds applied:")
    for name, (low, high) in zip(["gleak", "gklt", "gh"], [gleak_bounds, gklt_bounds, gh_bounds]):
        print(f" - {name:6s}: [{low:.2f}, {high:.2f}]")

    print("🔍 Running passive optimization...")
    result = minimize(compute_ess, initial, bounds=bounds)
    opt_leak, opt_gklt, opt_gh, opt_erev, opt_gkht, opt_gna, opt_gka = result.x

    # --- Apply best-fit params
    soma.g_leak = nstomho(opt_leak, somaarea)
    soma.gkltbar_LT_dth = nstomho(opt_gklt, somaarea)
    soma.ghbar_IH_nmb = nstomho(opt_gh, somaarea)
    soma.erev_leak = opt_erev
    soma.gkhtbar_HT_dth_nmb = nstomho(opt_gkht, somaarea)
    soma.gnabar_NaCh_nmb = nstomho(opt_gna, somaarea)
    soma.gkabar_ka = nstomho(opt_gka, somaarea)

    simulated_voltages_full = np.array([run_simulation(i) for i in all_currents])

    # --- Compute simulated voltages only for the fitted currents
    sim_fit = np.array([run_simulation(i) for i in fit_currents])

    # --- Fit-specific RMSE and R²
    rmse_fit = np.sqrt(mean_squared_error(fit_voltages, sim_fit))
    r2_fit = r2_score(fit_voltages, sim_fit)

    # --- Full-trace (all) RMSE and R² (already present)
    rmse_all = np.sqrt(mean_squared_error(all_steady_state_voltages, simulated_voltages_full))
    r2_all = r2_score(all_steady_state_voltages, simulated_voltages_full)

    residuals = all_steady_state_voltages - simulated_voltages_full
    rmse = np.sqrt(mean_squared_error(all_steady_state_voltages, simulated_voltages_full))
    r2 = r2_score(all_steady_state_voltages, simulated_voltages_full)

    # --- Save outputs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(script_dir, "..", "figures", f"fit_passive_{file_base}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    param_txt = os.path.join(script_dir, "..", "results", "_fit_results", f"passive_params_{file_base}_{timestamp}.txt")
    with open(param_txt, "w") as f:
        f.write(f"{opt_leak},{opt_gklt},{opt_gh},{opt_erev},{opt_gkht},{opt_gna},{opt_gka}\n")

    # legacy support
    with open(os.path.join(script_dir, "..", "results", "_fit_results", "best_fit_params.txt"), "w") as f:
        f.write(f"{opt_leak},{opt_gklt},{opt_gh},{opt_erev},{opt_gkht},{opt_gna},{opt_gka}\n")

    # --- Input resistance
    mask = (all_currents >= -0.020) & (all_currents <= 0.020)
    rin_exp = np.polyfit(all_currents[mask], all_steady_state_voltages[mask], 1)[0]
    rin_sim = np.polyfit(all_currents[mask], simulated_voltages_full[mask], 1)[0]

    # --- Summary JSON
    summary_path = os.path.join(output_dir, f"passive_summary_{file_base}.json")
    summary = {
        "gleak": opt_leak, "gklt": opt_gklt, "gh": opt_gh, "erev": opt_erev,
        "gkht": opt_gkht, "gna": opt_gna, "gka": opt_gka,
        "rin_exp_mohm": rin_exp,
        "rin_sim_mohm": rin_sim,
        "rmse_fit_mV": rmse_fit,
        "r2_fit": r2_fit,
        "rmse_all_mV": rmse_all,
        "r2_all": r2_all
    }

    print("\n📈 Fit Quality Metrics:")
    print(f"RMSE (fit points only): {rmse_fit:.2f} mV")
    print(f"R²   (fit points only): {r2_fit:.4f}")
    print(f"RMSE (all points):      {rmse_all:.2f} mV")
    print(f"R²   (all points):      {r2_all:.4f}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # --- Plot VI curve
    x_min = -0.15
    x_max = 0.4
    y_min = -110
    y_max = -20
    plt.figure(figsize=(8, 8))
    plt.scatter(all_currents, all_steady_state_voltages, color='r', label="Experimental")
    plt.plot(all_currents, simulated_voltages_full, '-', color='b', label="Simulated")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("Injected Current (nA)")
    plt.ylabel("Steady-State Voltage (mV)")
    plt.title(f"Passive Fit: Experimental vs Simulated {file_base}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "passive_fit.pdf"), dpi=300)
    plt.close()

    # --- Plot Input Resistance
    plt.figure(figsize=(8, 8))
    plt.plot(all_currents[mask], all_steady_state_voltages[mask], 'o', label="Exp")
    plt.plot(all_currents[mask], rin_exp * all_currents[mask] + np.mean(all_steady_state_voltages[mask]), '-', label=f"Exp Fit ({rin_exp:.2f} MΩ)")
    plt.plot(all_currents[mask], simulated_voltages_full[mask], 's', label="Sim")
    plt.plot(all_currents[mask], rin_sim * all_currents[mask] + np.mean(simulated_voltages_full[mask]), '--', label=f"Sim Fit ({rin_sim:.2f} MΩ)")
    plt.xlabel("Injected Current (nA)")
    plt.ylabel("Steady-State Voltage (mV)")
    plt.title("Input Resistance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_resistance_fit.png"), dpi=300)
    plt.close()

    # --- Plot Residual
    plt.figure(figsize=(8, 8))
    plt.bar(all_currents, residuals, width=0.01, color='purple', alpha=0.7)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Injected Current (nA)")
    plt.ylabel("Residual Voltage (mV)")
    plt.title("Residuals: Experimental - Simulated")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "residuals.pdf"), dpi=300)
    plt.close()

    print(f"✅ Passive fit complete: results saved to {output_dir}")
    print(f"⏱️ Elapsed: {time.time() - start_time:.2f} s")

    return PassiveParams(opt_leak, opt_gklt, opt_gh, opt_erev, opt_gkht, opt_gna, opt_gka), output_dir


# === Optional CLI ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to experimental CSV with columns 'Current', 'SteadyStateVoltage'")
    args = parser.parse_args()
    fit_passive(args.data)
