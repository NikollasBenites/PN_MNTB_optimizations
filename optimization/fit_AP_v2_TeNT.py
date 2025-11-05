import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, differential_evolution
from neuron import h
from collections import namedtuple
import MNTB_PN_myFunctions as mFun
import datetime
from MNTB_PN_fit import MNTB
from sklearn.metrics import mean_squared_error, r2_score
import time
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # TrueType
ParamSet = namedtuple("ParamSet", [
    "gna", "gkht", "gklt", "gh", "gka", "gleak", "stim_amp","cam","kam","cbm","kbm"
])

h.load_file('stdrun.hoc')
np.random.seed(42)
script_dir = os.path.dirname(os.path.abspath(__file__))

def fit_ap_tent(filename, stim_amp, param_file,batch_mode = False, expected_pattern = None):
    np.random.seed(42)
    print(f'Running AP fit for {filename}')
    print(f'stim_amp: {stim_amp}')
    param_file = os.path.join(script_dir, "..", "results", "_fit_results", "_latest_passive_fits", "TeNT",
                              param_file)
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Passive parameters not found at: {param_file}")
    with open(param_file, "r") as f:
        gleak, gklt, gh, erev, gkht, gna, gka = map(float, f.read().strip().split(","))

    stim_amp = float(stim_amp)
    # === Create Output Folder ===
    file = filename.split(".")[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(),"..", "results","_latest_iMNTB_TeNT_fits","_last_round","test_mutation","TeNT", f"fit_AP_{file}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    valid_patterns = ["phasic", "tonic", "silent", "non-phasic"]

    if batch_mode:
        expected_pattern = "tonic"
    else:
        try:
            expected_pattern = input(f"At +20 pA above rheobase, is the neuron {valid_patterns}? ").strip().lower()
        except EOFError:
            expected_pattern = "phasic"
    assert expected_pattern in valid_patterns, f"Please enter one of: {valid_patterns}"

    # Load experimental data
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "ap_P9_TeNT", filename))
    experimentalTrace = np.genfromtxt(data_path, delimiter=',', skip_header=1, dtype=float, filling_values=np.nan)

    t_exp = experimentalTrace[:,0] # ms
    t_exp = t_exp - t_exp[0]
    samples = np.sum((t_exp >= 0) & (t_exp <= 0.002))
    if samples > 2:
        t_exp = t_exp*1000
        print("t_exp converted to ms")

    V_exp = experimentalTrace[:,1]  # mV
    fp = V_exp[0]
    if abs(fp) < 1:
        V_exp *= 1000
        print("V_exp converted to mV")
    if not batch_mode:
        plt.figure(figsize=(10,8))
        plt.plot(t_exp,V_exp, linewidth=1.5)
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (mV)")
        plt.show(block=False)
        # === Compute sampling frequency from ms → Hz
    # fs = 1000 / (t_exp[1] - t_exp[0])  # Correct fs in Hz
    # V_exp = mFun.lowpass_filter(V_exp, cutoff=1000, fs=fs)
    # print(f"✅ Applied low-pass filter at 1 kHz (fs = {fs:.1f} Hz)")
    # Define soma parameters
    relaxation = 200
    totalcap = 25  # Total membrane capacitance in pF for the cell (input capacitance)
    somaarea = (totalcap * 1e-6) / 1  # pf -> uF,assumes 1 uF/cm2; result is in cm2
    threspass = 20 #dVdt pass –> threshold AP to simulated
    ek = -106.81
    ena = 62.77

    ################# sodium kinetics
    cam = 76.4 #76.4
    kam = .037
    cbm = 6.930852 #6.930852
    kbm = -.043

    cah = 0.000533  #( / ms)
    kah = -0.0909   #( / mV)
    cbh = 0.787     #( / ms)
    kbh = 0.0691    #( / mV)

    lbkna = 0.7
    hbkna = 1.3

    cell = MNTB(0,somaarea,erev,gleak,ena,gna,gh,gka,gklt,gkht,ek,cam,kam,cbm,kbm,cah,kah,cbh,kbh)

    stim_dur = 300
    stim_delay = 10

    lbamp = 0.999
    hbamp = 1.001

    # gleak = gleak
    lbleak = 0.999
    hbleak = 1.2

    gkht = 200
    lbKht = 0.5
    hbKht = 1.5

    if batch_mode:
        if gklt <= 5:
            gklt = gklt + 4
    else:
        if gklt <= 5:
            gklt = float(input(f"gKLT= {gklt}, what is the new value? "))

    lbKlt = 0.999
    hbKlt = 1.5

    gka = 100
    lbka = 0.1
    hbka = 1.5

    lbih = 0.999
    hbih = 1.001

    gna = 200
    lbgNa = 0.5
    hbgNa = 1.5

    bounds = [
        (gna*lbgNa, gna*hbgNa),             # gNa
        (gkht * lbKht, gkht * hbKht),
        (gklt * lbKlt, gklt * hbKlt),       # gKLT
        (gh * lbih, gh * hbih),             # gIH
        (gka * lbka, gka * hbka),           # gka
        (gleak * lbleak, gleak * hbleak),   # gleak
        (stim_amp*lbamp, stim_amp*hbamp),  # stim-amp
        (cam*lbkna, cam*hbkna),
        (kam*lbkna, kam*hbkna),
        (cbm*lbkna, cbm*hbkna),
        (kbm*hbkna, kbm*lbkna)
    ]

    def feature_cost(sim_trace, exp_trace, time, return_details=False):
        sim_feat = mFun.extract_features(sim_trace, time,threspass)
        exp_feat = mFun.extract_features(exp_trace, time,threspass=40)
        weights = {
            'rest':      1.0,
            'peak':      1.0,
            'amp':       1.0,
            'threshold': 1.0,
            'latency':   100.0,
            'width':     5.0,
            'AHP':       1.0
        }

        error = 0
        details = {}
        for k in weights:
            if not np.isnan(sim_feat[k]) and not np.isnan(exp_feat[k]):
                diff_sq = (sim_feat[k] - exp_feat[k])**2
                weighted = weights[k] * diff_sq
                error += weighted
                details[k] = {
                    'sim': sim_feat[k],
                    'exp': exp_feat[k],
                    'diff²': diff_sq,
                    'weighted_error': weighted
                }
            else:
                details[k] = {
                    'sim': sim_feat.get(k, np.nan),
                    'exp': exp_feat.get(k, np.nan),
                    'diff²': np.nan,
                    'weighted_error': np.nan
                }

        if return_details:
            return error, details
        else:
            return error


    #@lru_cache(maxsize=None)
    def run_simulation(p: ParamSet, stim_dur=300, stim_delay=10):
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
            "cam": p.cam,
            "kam": p.kam,
            "cbm": p.cbm,
            "kbm": p.kbm,
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
            stim_amp=p.stim_amp,
            stim_delay=stim_delay + relaxation,   #  ms extra relaxation
            stim_dur=stim_dur,
            v_init=v_init,
            total_duration=510 + relaxation,      # full  ms sim
            return_stim=False
        )

        return t, v


    # def interpolate_simulation(t_neuron, v_neuron, t_exp):
    #     interp_func = interp1d(t_neuron, v_neuron, kind='cubic', fill_value='extrapolate')
    #     return interp_func(t_exp)

    def interpolate_simulation(t_neuron, v_neuron, t_exp):
        # Ensure numpy arrays
        t_neuron = np.asarray(t_neuron)
        v_neuron = np.asarray(v_neuron)
        t_exp = np.asarray(t_exp)

        # Sort the time and voltage arrays (required for spline)
        sort_idx = np.argsort(t_neuron)
        t_neuron = t_neuron[sort_idx]
        v_neuron = v_neuron[sort_idx]

        # Create cubic spline with extrapolation enabled
        spline = CubicSpline(t_neuron, v_neuron, bc_type='natural', extrapolate=True)

        # Evaluate spline at experimental time points (t_exp)
        v_interp = spline(t_exp)

        return v_interp


    def penalty_terms(v_sim, dt=2e-5, expected_pattern = None):
        from scipy.signal import find_peaks

        peak = np.max(v_sim)
        rest = v_sim[0]
        penalty = 0

        # === Peak value penalty ===
        if peak < -15 or peak > 20:
            penalty += 1

        # === Resting potential penalty ===
        if rest > -55 or rest < -90:
            penalty += 1000

        # === Spike count penalty ===
        # Find peaks above a threshold (e.g., 0 mV) with minimum spacing to avoid noise
        refractory = max(1, int(1e-3 / dt))  # Ensure it's always >= 1
        peaks, _ = find_peaks(v_sim, height=0, distance=refractory)

        n_spikes = len(peaks)
        if expected_pattern == "phasic" and n_spikes > 1:
            penalty += 1000 * (n_spikes - 1)  # Penalize each extra spike

        return penalty

    def cost_function1(params):
        """
        Cost function for AP fitting. Compares interpolated simulation to experimental data
        based on MSE, feature differences, AP timing, and resting potential.
        """
        assert len(params) == 11, "Mismatch in number of parameters"
        p = ParamSet(*params)

        # === Run simulation ===
        t_sim, v_sim = run_simulation(p)

        if len(t_sim) < 2 or len(v_sim) < 2:
            print("[ERROR] Simulation returned insufficient points")
            return 1e6  # High cost to penalize failure

        # === Trim to match experimental time (0–510 ms) ===
        mask = t_sim >= relaxation
        t_sim = t_sim[mask] - relaxation  # Align simulation time to start at 0
        v_sim = v_sim[mask]

        if len(t_sim) < len(t_exp):
            print("[ERROR] Trimmed simulation too short")
            return 1e6

        # === Interpolate to experimental resolution ===
        try:
            v_interp = interpolate_simulation(t_sim, v_sim, t_exp)
        except Exception as e:
            print(f"[ERROR] Interpolation failed: {e}")
            return 1e6

        # === Feature extraction ===
        exp_feat = mFun.extract_features(V_exp, t_exp, threspass=35)
        sim_feat = mFun.extract_features(v_interp, t_exp, threspass)

        if np.isnan(exp_feat['latency']) or np.isnan(sim_feat['latency']):
            # print("[ERROR] No AP detected")
            return 1e6

        # === Define AP window ===
        dt = t_exp[1] - t_exp[0]
        ap_start = max(0, int((exp_feat['latency'] - 3) / dt))
        ap_end = min(len(t_exp), int((exp_feat['latency'] + 4) / dt))

        if ap_end <= ap_start:
            print("[ERROR] Invalid AP window")
            return 1e6

        v_interp_ap = v_interp[ap_start:ap_end]
        v_exp_ap = V_exp[ap_start:ap_end]
        t_ap = t_exp[ap_start:ap_end]

        # === Cost components core ===
        mse = np.mean((v_interp_ap - v_exp_ap) ** 2)
        f_cost = feature_cost(v_interp_ap, v_exp_ap, t_ap)
        time_shift = abs(np.argmax(v_interp_ap) - np.argmax(v_exp_ap)) * dt
        time_error = 500 * time_shift
        penalty = penalty_terms(v_sim, dt=dt, expected_pattern=expected_pattern)

        if expected_pattern == "phasic":
            # Create a new param set with stim_amp + 0.050 (50pA)
            p50 = p._replace(stim_amp=p.stim_amp + 0.050)
            t_sim_50, v_sim_50 = run_simulation(p50)

            if len(t_sim_50) > 10:
                v_spike_check = v_sim_50[t_sim_50 >= relaxation]
                from scipy.signal import find_peaks
                refractory = max(1, int(1e-3 / dt))  # Ensure it's always >= 1
                peaks, _ = find_peaks(v_sim, height=0, distance=refractory)

                peaks_50, _ = find_peaks(v_spike_check, height=0, distance=refractory)
                n_spikes_50 = len(peaks_50)

                if n_spikes_50 > 1:
                    penalty += 1000 * (n_spikes_50 - 1)

        elif expected_pattern == "tonic":
            p20 = p._replace(stim_amp=p.stim_amp + 0.020)
            t_sim_20, v_sim_20 = run_simulation(p20)
            if len(t_sim_20) > 10:
                v_spike_check = v_sim_20[t_sim_20 >= relaxation]
                from scipy.signal import find_peaks
                refractory = max(1, int(1e-3 / dt))  # Ensure it's always >= 1
                peaks, _ = find_peaks(v_sim, height=0, distance=refractory)

                peaks_20, _ = find_peaks(v_spike_check, height=0, distance=refractory)
                n_spikes_20 = len(peaks_20)

                if n_spikes_20 > 60:
                    penalty += 1000 * (n_spikes_20 - 60)

        # === Total weighted cost ===
        alpha = 1  # MSE weight
        beta = 1  # Feature cost weight


        total_cost = alpha * mse + beta * f_cost + time_error + penalty
        # if np.random.rand() < 0.01:
        #     plt.figure()
        #     plt.plot(t_exp, V_exp, label='Exp', alpha=0.7)
        #     plt.plot(t_exp, v_interp, label='Sim', alpha=0.7)
        #     plt.legend()
        #     plt.title(f"Trace Comparison | Cost: {total_cost:.2f}")
        #     plt.show()

        return total_cost


    def log_and_plot_optimization(result_global, result_local, param_names=None, save_path=None):
        """
        Compare and plot cost function results from global and local optimizations.

        Parameters:
        - result_global: output from differential_evolution
        - result_local: output from minimize
        - param_names: list of parameter names (optional)
        - save_path: if provided, save log CSV and plot there
        """
        global_cost = result_global.fun
        local_cost = result_local.fun
        delta = global_cost - local_cost

        print(f"🔍 Global Cost: {global_cost:.6f}")
        print(f"🔧 Local Cost:  {local_cost:.6f}")
        print(f"📉 Improvement: {delta:.6f}")

        # Combine parameter sets
        if param_names is None:
            param_names = [f'p{i}' for i in range(len(result_local.x))]

        df = pd.DataFrame({
            'Parameter': param_names,
            'Global_fit': result_global.x,
            'Local_fit': result_local.x
        })
        df['Change'] = df['Local_fit'] - df['Global_fit']

        print("\nParameter changes:\n", df)

        # Optional save
        if save_path:
            csv_path = f"{save_path}/fit_log.csv"
            df.to_csv(csv_path, index=False)
            print(f"\n✅ Saved log to {csv_path}")

        # Optional bar plot of change
        plt.figure(figsize=(8, 4))
        plt.bar(df['Parameter'], df['Change'], color='skyblue')
        plt.axhline(0, linestyle='--', color='gray')
        plt.ylabel("Change (Local - Global)")
        plt.title("Parameter Adjustment After Local Fit")
        plt.tight_layout()

        if save_path:
            fig_path = f"{save_path}/fit_comparison_plot.png"
            plt.savefig(fig_path, dpi=300)
            print(f"📊 Saved plot to {fig_path}")
        else:
            if not batch_mode:
                plt.show()

    def create_local_bounds(center, rel_window=0.1, abs_min=None, abs_max=None):
        """Create a tuple (min, max) around center using ±rel_window, ensuring valid order even for negative center."""
        lower = center * (1 - rel_window)
        upper = center * (1 + rel_window)

        # Ensure correct order
        bound_min, bound_max = min(lower, upper), max(lower, upper)

        # Clamp to absolute bounds if given
        if abs_min is not None:
            bound_min = max(bound_min, abs_min)
        if abs_max is not None:
            bound_max = min(bound_max, abs_max)

        return (bound_min, bound_max)

    print(f"gKLT: {gklt}")
    print("Running optimization...")
    ti = time.time()
    t0 = time.time()
    result_global = differential_evolution(cost_function1, bounds, strategy='best1bin', maxiter=20,
                                           updating='immediate' ,popsize=100, mutation=0.5,recombination=0.1, polish=True, tol=1e-4)
    t1 = time.time()
    print(f"✅ Global optimization done in {t1 - t0:.2f} seconds")
    print("Running minimization...")
    t2 = time.time()
    result_local = minimize(cost_function1, result_global.x, bounds=bounds, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True})
    t3 = time.time()
    print(f"✅ Local minimization done in {t3 - t2:.2f} seconds")
    print(f"🕒 Total optimization time: {t3 - t0:.2f} seconds")

    def run_refinement_loop(initial_result, cost_func, rel_windows, max_iters=150, min_delta=1e-6, max_restarts=5):
        history = [initial_result.fun]
        current_result = initial_result

        print("\n🔁 Starting refinement loop:")
        restarts = 0

        while True:
            converged = False

            for i in range(max_iters):
                print(f"\n🔂 Iteration {i+1} (Restart {restarts})")

                x_opt = current_result.x
                new_bounds = [
                    create_local_bounds(x_opt[j], rel_window=rel_windows[j])
                    for j in range(len(x_opt))
                ]

                new_result = differential_evolution(
                    cost_func,
                    bounds=new_bounds,
                    strategy='best1bin',
                    maxiter=20,
                    updating='immediate',
                    popsize=100,
                    mutation=0.5,
                    recombination=0.1,
                    polish=True,
                    tol=1e-4
                )

                delta = current_result.fun - new_result.fun
                history.append(new_result.fun)

                print(f"   Cost: {current_result.fun:.4f} → {new_result.fun:.6f} (Δ = {delta:.6f})")

                if delta < min_delta:
                    print("   ✅ Converged: small improvement.")
                    converged = True
                    break

                current_result = new_result

            if converged or restarts >= max_restarts:
                break

            print("🔁 Restarting refinement loop (did not converge yet)...")
            restarts += 1

        return current_result, history


    def count_spikes(trace, time, threshold=-15):
        """
        Count number of spikes based on upward threshold crossings.
        """
        above = trace > threshold
        crossings = np.where(np.diff(above.astype(int)) == 1)[0]
        return len(crossings)
    def classify_firing_pattern(n_spikes):
        if n_spikes == 0:
            return "silent"
        elif n_spikes == 1:
            return "phasic"
        elif n_spikes >= 4:
            return "tonic"
        else:
            return "non-phasic"
    def verify_rheobase_fit(
        params_opt, t_exp, V_exp,
        threshold_mse=1.0,
        threshold_r2=0.85,
        threshold_time_shift=1.0,
        threspass=20,
        verbose=False
    ):
        """
        Simulate the model at rheobase and verify if it still fits well.
        Returns: (is_valid: bool, t_rheo, v_rheo)
        """
        from sklearn.metrics import mean_squared_error, r2_score

        t_rheo, v_rheo = run_simulation(params_opt)
        v_rheo_interp = interpolate_simulation(t_rheo, v_rheo, t_exp)

        mse = mean_squared_error(V_exp, v_rheo_interp)
        r2 = r2_score(V_exp, v_rheo_interp)
        feature_err = feature_cost(v_rheo_interp, V_exp, t_exp)
        time_shift = abs(np.argmax(v_rheo_interp) - np.argmax(V_exp)) * (t_exp[1] - t_exp[0])

        is_valid = (mse < threshold_mse) and (r2 > threshold_r2) and (time_shift < threshold_time_shift)

        if verbose:
            print(f"\n📈 Rechecking rheobase fit:")
            print(f"→ MSE: {mse:.4f}")
            print(f"→ R²:  {r2:.4f}")
            print(f"→ Time shift: {time_shift:.2f} ms")
            print(f"→ Feature Error: {feature_err:.2f}")
            print("✅ Rheobase fit is acceptable." if is_valid else "❌ Rheobase fit is degraded.")

        return is_valid, t_rheo, v_rheo
    def check_and_refit_if_needed(
        params_opt, expected_pattern, t_exp, V_exp, rel_windows,
        output_dir, max_retries=10, do_refit=None, fixed_params: list[str] = None
    ):
        def simulate_plus_20(p):
            stim_amp_plus_50 = p.stim_amp + 0.020
            test_p = p._replace(stim_amp=stim_amp_plus_50)
            t_hi, v_hi = run_simulation(test_p)
            n_spikes = count_spikes(v_hi, t_hi)
            pattern = classify_firing_pattern(n_spikes)
            return pattern, n_spikes, t_hi, v_hi

        print(f"\n🔍 Verifying +20 pA response:")
        pattern, n_spikes, t_hi, v_hi = simulate_plus_20(params_opt)
        print(f"Expected: {expected_pattern}, Observed: {pattern} ({n_spikes} spike{'s' if n_spikes != 1 else ''})")

        if pattern == expected_pattern:
            print("✅ Match confirmed. No re-optimization needed.")
            return params_opt, False, t_hi, v_hi, pattern

        # === Ask user only if do_refit was not passed
        if do_refit is None:
            try:
                user_input = input("Neuron didn't fit the pattern. Do you want to refit? (y/n): ").strip().lower()
            except EOFError:
                user_input = 'n'
            do_refit = user_input == 'y'

        if not do_refit:
            print("⚠️  Not phasic, but re-optimization is skipped.")
            return params_opt, False, t_hi, v_hi, pattern

        print("❌ Not phasic. Re-optimizing selected channels...")

        fixed_dict = params_opt._asdict().copy()
        all_param_names = ['gna', 'gkht', 'gka', 'gklt']
        fixed_params = fixed_params or []
        param_names = [p for p in all_param_names if p not in fixed_params]

        x0 = [fixed_dict[k] for k in param_names]
        fixed = {k: v for k, v in fixed_dict.items() if k in fixed_params or k not in all_param_names}

        retries = 0
        new_params = params_opt
        successful = False

        while retries < max_retries:
            initial_scale = 0.5
            decay = 0.95 ** retries

            broader_bounds = []
            for pname in param_names:
                val = fixed_dict[pname]
                if pname in ["gna", "gkht"]:
                    lower = val * initial_scale * decay
                    upper = max(lower + 1e-9, val * (1.0 - 0.1 * retries))
                elif pname in ["gklt","gka"]:
                    lower = val * (1.0 + 0.02 * retries)
                    upper = val * (1.0 + 0.04 * retries)
                else:
                    continue
                broader_bounds.append((lower, upper))
                print(f"Retry {retries}: {pname}_bound = ({lower:.4g}, {upper:.4g})")

            def cost_partial(x):
                pdict = fixed.copy()
                pdict.update(dict(zip(param_names, x)))
                return cost_function1(ParamSet(**pdict))

            result_global = differential_evolution(
                cost_partial, broader_bounds, strategy='best1bin',
                maxiter=10, popsize=50, mutation=1.0,
                updating='immediate', polish=False, tol=1e-4
            )

            result_local = minimize(
                cost_partial, result_global.x, bounds=broader_bounds,
                method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-6, 'disp': True}
            )

            updated = fixed.copy()
            updated.update(dict(zip(param_names, result_local.x)))
            new_params = ParamSet(**updated)

            # Check +20 pA
            pattern, n_spikes, t_hi, v_hi = simulate_plus_20(new_params)
            print(f"   → Observed: {pattern.upper()} with {n_spikes} spike(s)")

            if pattern == "phasic":
                print("🎯 Achieved phasic firing. Verifying rheobase fit...")
                is_valid, t_rheo, v_rheo = verify_rheobase_fit(new_params, t_exp, V_exp)

                if is_valid:
                    print("✅ Rheobase fit is acceptable. Finalizing.")
                    successful = True
                    break
                else:
                    print("❌ Rheobase fit degraded. Retrying...")

            retries += 1

        if not successful:
            print("⚠️ Could not achieve valid phasic behavior and rheobase fit after retries.")
            t_rheo, v_rheo = run_simulation(new_params)  # last attempt for export

        # === Save summary
        summary_path = os.path.join(output_dir, "refit_summary.json")
        summary = {
            "reoptimized": True,
            "target": "phasic",
            "final_pattern": pattern,
            "final_spikes": n_spikes,
            "rheobase_fit_valid": successful,
            "n_retries": retries,
            "stim_amp_plus_50": new_params.stim_amp + 0.050
        }
        with open(summary_path, "w") as f:
            import json
            json.dump(summary, f, indent=4)
        print(f"📝 Saved refit summary to {summary_path}")

        return new_params, True, t_hi, v_hi, pattern

    rel_windows = [
        0.1,  # gNa: sodium conductance — narrow ±10%
        0.1,  # gKHT: high-threshold K⁺ conductance — broader ±50%
        0.01,  # gKLT: low-threshold K⁺ conductance — broader ±50%
        0.01,  # gIH: HCN conductance — narrow ±10%
        0.1,  # gKA: A-type K⁺ conductance — broader ±50%
        0.01,  # gLeak: leak conductance — narrow ±0.1%
        0.0,  # stim_amp: current amplitude — narrow ±50%
        0.1,  # cam: Na⁺ activation slope — narrow ±10%
        0.1,  # kam: Na⁺ activation V-half — narrow ±10%
        0.1,  # cbm: Na⁺ inactivation slope — narrow ±10%
        0.1   # kbm: Na⁺ inactivation V-half — narrow ±10%
    ]


    result_local_refined, cost_history = run_refinement_loop(result_local, cost_function1, rel_windows)

    params_opt = ParamSet(*result_local_refined.x)
    print("Optimized parameters:")
    for name, value in params_opt._asdict().items():
        print(f"{name}: {value:.4f}")

    params_opt, reoptimized, t_hi, v_hi, pattern = check_and_refit_if_needed(
        params_opt, expected_pattern, t_exp, V_exp, rel_windows, output_dir, max_retries=5,fixed_params=['gka','gkht','gklt'],do_refit=False
    )
    param_names = ParamSet._fields

    log_and_plot_optimization(result_global, result_local, param_names, save_path=os.path.join(output_dir))
    log_and_plot_optimization(result_local, result_local_refined, param_names,save_path=os.path.join(output_dir))
    #
    print(f"Best stim-amp: {params_opt.stim_amp:.2f} pA")
    print(f" Optimized gna: {params_opt.gna:.2f}, gklt: {params_opt.gklt: .2f}, gkht: {params_opt.gkht: .2f}), gh: {params_opt.gh:.2f}, gka:{params_opt.gka:.2f}, gleak: {params_opt.gleak:.2f}, "
          f"cam: {params_opt.cam: .2f}, kam: {params_opt.kam: .2f}, cbm: {params_opt.cbm: .2f}, kbm: {params_opt.kbm: .2f}")

    # Final simulation and plot
    t_sim, v_sim = run_simulation(params_opt)

    tf = time.time()

    print(f"Whole Simulation took {tf-ti:.2f} seconds")
    # === Trim simulation to remove 200 ms buffer for proper plotting
    t_trimmed = t_sim[t_sim >= relaxation] - relaxation
    v_trimmed = v_sim[t_sim >= relaxation]
    # Interpolate simulated trace to match experimental time points
    v_interp = interpolate_simulation(t_trimmed, v_trimmed, t_exp)

    feat_sim = mFun.extract_features(v_trimmed, t_trimmed,threspass)
    print("Simulate Features:")
    for k, v in feat_sim.items():
        print(f"{k}: {v:.2f}")

    feat_exp = mFun.extract_features(V_exp,t_exp,threspass=35)
    print("Experimental Features:")
    for k, v in feat_exp.items():
        print(f"{k}: {v:.2f}")

    results = params_opt._asdict()
    results.update(feat_sim)

    combined_results = {
        "gleak": gleak, "gklt": params_opt.gklt, "gh": params_opt.gh, "erev": erev,
        "gna": params_opt.gna, "gkht": params_opt.gkht, "gka": params_opt.gka,
        "stim_amp": params_opt.stim_amp,
        "cam": params_opt.cam, "kam": params_opt.kam, "cbm": params_opt.cbm, "kbm": params_opt.kbm
    }

    pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "..","results","_fit_results", f"all_fitted_params_{file}_{timestamp}.csv"), index=False)
    pd.DataFrame([combined_results]).to_csv(os.path.join(script_dir, "..","results","_fit_results", f"all_fitted_params.csv"), index=False) #the last
    # monitor_cache_size()

    plt.figure(figsize=(10, 5))
    plt.plot(t_exp, V_exp, label='Experimental', linewidth=2)
    plt.plot(t_trimmed, v_trimmed, label='Simulated (fit)', linestyle='--')

    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('Action Potential Fit')
    thresh_exp = mFun.extract_features(V_exp, t_exp,threspass=35)['latency']
    thresh_sim = mFun.extract_features(v_trimmed, t_trimmed, threspass)['latency']

    plt.axvline(thresh_exp, color='blue', linestyle=':', label='Exp Threshold')
    plt.axvline(thresh_sim, color='orange', linestyle=':', label='Sim Threshold')
    plt.tight_layout()
    if batch_mode:
        ap_exp_sim = os.path.join(output_dir,"ap_exp_sim_rheobase.png")
        plt.savefig(ap_exp_sim, dpi=300,bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t_hi, v_hi, label='Simulated (fit) +20 pA', linewidth=2)
    plt.legend()
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane potential (mV)')
    plt.title('Action Potential Fit')
    plt.tight_layout()
    if batch_mode:
        ap_exp_sim_20 = os.path.join(output_dir, "ap_exp_sim_20pA.png")
        plt.savefig(ap_exp_sim_20, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # half_amp = feat_sim['threshold'] + 0.5 * feat_sim['amp']
    # plt.plot(t_sim, v_sim)
    # plt.axhline(half_amp, color='red', linestyle='--', label='Half amplitude')
    # plt.title("Check AP width region")
    # plt.legend()
    # plt.show()

    # Save +50 pA trace
    trace_df = pd.DataFrame({
        "time_ms": t_hi,
        "voltage_mV": v_hi
    })
    trace_file = os.path.join(output_dir, f"sim_trace_plus_50pA_{pattern}.csv")
    trace_df.to_csv(trace_file, index=False)
    print(f"💾 Saved +50 pA trace to {trace_file}")

    # === Plot clipped AP window ===
    dt = t_exp[1] - t_exp[0]
    try:
        ap_start = max(0, int((feat_exp['latency'] - 3) / dt))
        ap_end = min(len(t_exp), int((feat_exp['latency'] + 4) / dt))
    except Exception as e:
        print("⚠️ Could not clip AP window:", e)
        ap_start = 0
        ap_end = len(t_exp)

    t_clip = t_exp[ap_start:ap_end]
    v_exp_clip = V_exp[ap_start:ap_end]
    v_sim_clip = interpolate_simulation(t_trimmed, v_trimmed, t_clip)

    plt.figure(figsize=(8, 4))
    plt.plot(t_clip, v_exp_clip, label='Experimental (filtered)', linewidth=2)
    plt.plot(t_clip, v_sim_clip, '--', label='Simulated (fit)', linewidth=2)
    plt.xlabel("Time (ms)")
    plt.ylabel("Membrane potential (mV)")
    plt.title("AP Fit — Clipped to AP Window")
    plt.legend()
    plt.tight_layout()
    if batch_mode:
        ap_exp_sim_window = os.path.join(output_dir, "ap_exp_sim_window.png")
        plt.savefig(ap_exp_sim_window, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Compute fit quality metrics
    # === Compute metrics in the AP window only ===
    mse = mean_squared_error(v_exp_clip, v_sim_clip)
    r2 = r2_score(v_exp_clip, v_sim_clip)
    time_shift = abs(np.argmax(v_sim_clip) - np.argmax(v_exp_clip)) * (t_exp[1] - t_exp[0])
    feature_error = feature_cost(v_sim_clip, v_exp_clip, t_clip)

    # Add fit quality label
    fit_quality = 'good' if r2 > 0.9 and time_shift < 0.5 else 'poor'
    results['mse'] = mse
    results['r2'] = r2
    results['time_shift'] = time_shift
    results['feature_error'] = feature_error
    results['fit_quality'] = fit_quality
    print(f"\n=== Fit Quality Metrics ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R-squared (R²):           {r2:.4f}")
    print(f"Time Shift (ms):          {time_shift:.4f}")
    print(f"Feature Error:            {feature_error:.2f}")
    print(f"Fit Quality:              {fit_quality}")

    f_error, f_details = feature_cost(v_interp, V_exp, t_exp, return_details=True)

    print("\n📊 Feature-wise Error Breakdown:")
    for feat, vals in f_details.items():
        print(f"{feat:10s} | Sim: {vals['sim']:.2f} | Exp: {vals['exp']:.2f} | Diff²: {vals['diff²']:.2f} | Weighted: {vals['weighted_error']:.2f}")

    pd.DataFrame([results]).to_csv(os.path.join(output_dir,f"fit_results_{timestamp}.csv"), index=False)

    results_exp = {k: v for k, v in results.items() if k in feat_exp}

    df = pd.DataFrame([results_exp])  # Create DataFrame first
    df = pd.DataFrame([results_exp]).to_csv(os.path.join(output_dir,f"fit_results_exp_{timestamp}.csv"), index=False)
    # === Save experimental and simulated AP window traces together ===
    ap_window_dir = os.path.join(output_dir, "ap_window_traces")
    os.makedirs(ap_window_dir, exist_ok=True)

    # Choose time base (aligned or not)
    # If you want to export aligned traces, use these:
    # time_base = t_clip_aligned_exp
    # v_exp_to_save = v_exp_aligned
    # v_sim_to_save = v_sim_aligned

    # If you want to export non-aligned traces, use these:
    time_base = t_clip
    v_exp_to_save = v_exp_clip
    v_sim_to_save = v_sim_clip

    # Save combined DataFrame
    combined_df = pd.DataFrame({
        "time_ms": time_base,
        "exp_voltage_mV": v_exp_to_save,
        "sim_voltage_mV": v_sim_to_save
    })

    combined_path = os.path.join(ap_window_dir, f"combined_ap_window_{filename}_{timestamp}.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"💾 Saved combined AP window to {combined_path}")


    # === Align by threshold (time and voltage) ===
    thresh_exp_time = feat_exp['latency']
    thresh_sim_time = feat_sim['latency']
    thresh_exp_voltage = feat_exp['threshold']
    thresh_sim_voltage = feat_sim['threshold']

    # Time alignment: shift time vectors so threshold is at 0
    t_clip_aligned_exp = t_clip - thresh_exp_time
    t_clip_aligned_sim = t_clip - thresh_sim_time

    # Voltage alignment: subtract threshold voltage so both traces start at 0 mV
    v_exp_aligned = v_exp_clip - thresh_exp_voltage
    v_sim_aligned = v_sim_clip - thresh_sim_voltage

    # === Plot aligned APs
    plt.figure(figsize=(8, 4))
    plt.plot(t_clip_aligned_exp, v_exp_aligned, label='Experimental (aligned)', linewidth=2)
    plt.plot(t_clip_aligned_sim, v_sim_aligned, '--', label='Simulated (aligned)', linewidth=2)
    plt.axvline(0, color='gray', linestyle=':', label='Threshold time')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel("Time (ms, aligned to threshold)")
    plt.ylabel("Membrane potential (mV, threshold-normalized)")
    plt.title("AP Fit — Aligned by Threshold Time and Voltage")
    plt.legend()
    plt.tight_layout()
    # Save to PDF
    aligned_pdf_path = os.path.join(output_dir, f"aligned_AP_fit_{filename}_{timestamp}.pdf")
    plt.savefig(aligned_pdf_path, format='pdf')
    print(f"📄 Saved aligned AP plot to: {aligned_pdf_path}")
    if batch_mode:
        plt.close()
    else:
        plt.show()

    return ParamSet, output_dir

# === Optional CLI ===
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", required=True, help="AP file to fit")
    parser.add_argument("--stim_amp", required=True, type=float, help="Rheobase amplitude in nA") #must be in the NEURON format nA
    parser.add_argument("--param_file", required=True, help="Passive parameters CSV file")
    parser.add_argument("--batch_mode", required=False, default=False, help="Batch mode")
    parser.add_argument("--expected_pattern", required=False, default=False, help="Expected pattern")
    args = parser.parse_args()

    fit_ap_tent(args.filename, args.stim_amp, args.param_file, args.batch_mode, args.expected_pattern)
