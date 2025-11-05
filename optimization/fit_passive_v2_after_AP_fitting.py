import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from neuron import h
from collections import namedtuple
import MNTB_PN_myFunctions as mFun
import datetime

PassiveParams = namedtuple("refinedParams", ["gleak", "gklt", "gh", "erev", "gka"])

def nstomho(x, somaarea):
    return (1e-9 * x / somaarea)

def fit_passive(data_filename, param_filename):
    start_time = datetime.datetime.now()

    # Load params
    param_df = pd.read_csv(param_filename)
    p = param_df.iloc[0]
    gna = p.get("gna", None)
    gkht = p.get("gkht", None)
    fixed_gka = p.get("gka", None)

    # Load data
    data = pd.read_csv(data_filename)
    currents = data["Stimulus"].values * 1e-3
    voltages = data["SteadyState (nA or mV)"].values

    # NEURON setup
    h.load_file("stdrun.hoc")
    h.celsius = 35
    v_init = -70

    somaarea = (25e-6 / 1)  # 25 pF

    def run_model(gleak, gklt, gh, erev, gka, gna=None, gkht=None):
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

        soma.g_leak = nstomho(gleak, somaarea)
        soma.gkltbar_LT_dth = nstomho(gklt, somaarea)
        soma.ghbar_IH_nmb = nstomho(gh, somaarea)
        soma.erev_leak = erev
        soma.gkabar_ka = nstomho(gka, somaarea)

        if gna is not None:
            soma.gnabar_NaCh_nmb = nstomho(gna, somaarea)
        if gkht is not None:
            soma.gkhtbar_HT_dth_nmb = nstomho(gkht, somaarea)

        stim = h.IClamp(soma(0.5))
        stim.dur = 300
        stim.delay = 10
        h.dt = 0.02

        v_vec = h.Vector().record(soma(0.5)._ref_v)
        t_vec = h.Vector().record(h._ref_t)

        def run(current):
            v_vec.resize(0)
            t_vec.resize(0)
            stim.amp = current
            h.v_init = v_init
            mFun.custom_init(v_init)
            h.tstop = stim.delay + stim.dur
            h.continuerun(h.tstop)
            t = np.array(t_vec)
            v = np.array(v_vec)
            return np.mean(v[(t >= 250) & (t <= 290)])

        return np.array([run(i) for i in currents])

    def compute_ess(params):
        gleak, gklt, gh, erev = params
        sim = run_model(
            gleak, gklt, gh, erev, fixed_gka,
            gna=gna, gkht=gkht
        )
        return np.sum((voltages - sim) ** 2)

    initial = [p["gleak"], p["gklt"], p["gh"], p["erev"]]
    bounds = [
        (v * 0.7, v * 1.3) if i != 3 else (v - 2.0, v + 2.0)
        for i, v in enumerate(initial)
    ]

    result = minimize(compute_ess, initial, bounds=bounds, method='L-BFGS-B')
    gleak, gklt, gh, erev = result.x
    gka = fixed_gka

    simulated = run_model(gleak, gklt, gh, erev, gka, gna=gna, gkht=gkht)

    rmse = np.sqrt(mean_squared_error(voltages, simulated))
    r2 = r2_score(voltages, simulated)

    print(f"\n✅ {os.path.basename(data_filename)}")
    print(f"RMSE: {rmse:.2f} mV\tR²: {r2:.4f}")

    # Save figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(os.path.dirname(__file__), "..", "figures", f"_recheck_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.plot(currents, voltages, 'ro', label="Experimental")
    plt.plot(currents, simulated, 'b-', label="Simulated")
    plt.xlabel("Current (nA)")
    plt.ylabel("Steady Vm (mV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{os.path.basename(data_filename).replace('.csv','')}_fitcheck.pdf"))
    plt.close()

    # Save results to CSV
    log_path = os.path.join(outdir, "refined_passive_params.csv")
    pd.DataFrame([{
        "filename": os.path.basename(data_filename),
        "gleak": gleak,
        "gklt": gklt,
        "gh": gh,
        "erev": erev,
        "gka": gka,
        "gna": gna,
        "gkht": gkht,
        "rmse": rmse,
        "r2": r2
    }]).to_csv(log_path, index=False)

    return PassiveParams(gleak, gklt, gh, erev, gka), outdir

# === Optional CLI ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to experimental CSV with 'Stimulus' and 'SteadyState (nA or mV)' columns")
    parser.add_argument("--params", required=True, help="Path to CSV with fitted passive parameters")
    args = parser.parse_args()
    fit_passive(args.data, args.params)
