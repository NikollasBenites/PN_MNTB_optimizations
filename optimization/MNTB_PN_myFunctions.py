import neuron
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from neuron import h
from scipy.signal import butter, filtfilt


h.load_file("stdrun.hoc")

def custom_init(v_init=-75):
    """
    Perform a custom initialization of the current model/section.

    This initialization follows the scheme outlined in the
    NEURON book, 8.4.2, p 197 for initializing to steady state.

    N.B.: For a complex model with dendrites/axons and different channels,
    this initialization will not find the steady-state the whole cell,
    leading to initial transient currents. In that case, this initialization
    should be followed with a 0.1-5 second run (depends on the rates in the
    various channel mechanisms) with no current injection or active point
    processes to allow the system to settle to a steady- state. Use either
    h.svstate or h.batch_save to save states and restore them. Batch is
    preferred

    Parameters
    ----------
    v_init : float (default: -60 mV)
        Voltage to start the initialization process. This should
        be close to the expected resting state.
    """
    inittime = -1e12
    tdt = neuron.h.dt  # save current step size
    dtstep = 1e9
    neuron.h.finitialize(v_init)
    neuron.h.t = inittime  # set time to large negative value (avoid activating
    # point processes, we hope)
    tmp = neuron.h.cvode.active()  # check state of variable step integrator
    if tmp != 0:  # turn off CVode variable step integrator if it was active
        neuron.h.cvode.active(0)  # now just use backward Euler with large step
    neuron.h.dt = dtstep
    n = 0
    while neuron.h.t < -1e9:  # Step forward
        neuron.h.fadvance()
        n += 1
    # print('advances: ', n)
    if tmp != 0:
        neuron.h.cvode.active(1)  # restore integrator
    neuron.h.t = 0
    if neuron.h.cvode.active():
        neuron.h.cvode.re_init()  # update d(state)/dt and currents
    else:
        neuron.h.fcurrent()  # recalculate currents
    neuron.h.frecord_init()  # save new state variables
    neuron.h.dt = tdt  # restore original time step

    return v_init

def run_simulation(amp, stim, soma_v, t, totalrun, stimdelay=None, stimdur=None, stim_traces=None, v_init=-70):
    # Set initial voltage
    h.v_init = v_init
    custom_init(v_init)

    # Set stimulation parameters
    stim.amp = amp
    if stimdelay is not None:
        stim.delay = stimdelay
    if stimdur is not None:
        stim.dur = stimdur

    # Run the simulation
    h.tstop = totalrun
    h.continuerun(totalrun)

    # Extract recorded values
    soma_values = np.array(soma_v.to_python())
    t_values = np.array(t.to_python())
    if stim_traces is not None:
        stim_values = np.array(stim_traces.to_python())
        return soma_values, stim_values, t_values
    else:
        return soma_values, t_values


def avg_ss_values(soma_values, t_values, t_min, t_max, average_soma_values):
    """
    Extract the range of soma_values based on t_values within the specified time range,
    calculate the average soma value within that range, and update the average soma values array.

    Parameters:
    - soma_values (np.array): Array of soma voltage values.
    - t_values (np.array): Array of time values.
    - t_min (float): Minimum time for the range.
    - t_max (float): Maximum time for the range.
    - average_soma_values (np.array): Array to store the average soma values.

    Returns:
    - tuple: Three elements containing the soma values within the specified range,
             the time values within the specified range, and the updated array of average soma values.
    """
    # Find the indices where t_values are within the specified range
    range_indices = np.where((t_values >= t_min) & (t_values <= t_max))

    # Extract and round the soma values and time values within the range
    soma_values_range = np.round(soma_values[range_indices], 3)
    t_values_range = np.round(t_values[range_indices], 3)

    # Calculate the average soma value within the range
    average_soma_value = np.round(np.mean(soma_values_range), 3)

    # Append the average soma value to the average_soma_values array
    average_soma_values = np.append(average_soma_values, average_soma_value)

    return soma_values_range, t_values_range, average_soma_values

def count_spikes(num_spikes,stimdelay,stimdur,spike_times,ap_counts,ap_times):
    num_spikes = sum(stimdelay <= time <= stimdelay + stimdur for time in spike_times)
    ap_counts.append(num_spikes)
    ap_times.append(list(spike_times))
    return num_spikes,spike_times,ap_counts, ap_times

def analyze_AP(time, voltage):
    """Analyze AP features from a single voltage trace."""
    stimdelay = 101

    # Compute first derivative
    dv_dt = np.gradient(voltage, time)

    # Detect peaks (APs)
    peaks, _ = find_peaks(voltage, height=-15)  # Adjust threshold if needed
    if len(peaks) == 0:
        return None  # No AP detected

    first_ap_idx = peaks[0]  # Index of first AP
    spike_time = time[first_ap_idx]
    spike_latency = spike_time - stimdelay

    # Limit threshold search to 5 ms before the AP peak
    search_start_time = spike_time - 1  # in ms
    if search_start_time < time[0]:
        search_start_time = time[0]

    search_start_idx = np.where(time >= search_start_time)[0][0]
    search_end_idx = first_ap_idx

    threshold_candidates = np.where(dv_dt[search_start_idx:search_end_idx] > 10)[0]
    if len(threshold_candidates) > 0:
        threshold_idx = search_start_idx + threshold_candidates[0]
    else:
        threshold_idx = first_ap_idx  # Fallback

    threshold_voltage = voltage[threshold_idx]
    peak_voltage = voltage[first_ap_idx]
    ap_amplitude = peak_voltage - threshold_voltage

    # Compute AP half-width
    half_max = threshold_voltage + ap_amplitude / 2
    crossings = np.where(voltage[:first_ap_idx] >= half_max)[0]
    if len(crossings) > 1:
        half_width = time[crossings[-1]] - time[crossings[0]]
    else:
        half_width = np.nan  # Could not determine half-width

    # Afterhyperpolarization (AHP) voltage
    ahp_voltage = np.min(voltage[first_ap_idx:])  # Minimum after AP

    return {
        "threshold": threshold_voltage,
        "peak": peak_voltage,
        "amplitude": ap_amplitude,
        "halfwidth": half_width,
        "AHP": ahp_voltage,
        "spike latency": spike_latency,
        "spike time": spike_time
    }

def compute_ess(params, soma, nstomho, somaarea, exp_currents, exp_steady_state_voltages,
                st, t_vec, v_vec, tmin=250, tmax=300):
    """
    Compute the explained sum of squares (ESS) for optimization.

    Parameters:
        params: list of [gleak, gka, gh, erev]
        soma: NEURON soma section
        nstomho: conversion function
        somaarea: area in cm2
        exp_currents: injected current steps (nA)
        exp_steady_state_voltages: target voltages for each step
        st: IClamp object
        t_vec, v_vec: h.Vector objects for recording time and voltage
        tmin, tmax: steady-state window (ms)

    Returns:
        ESS (float): error metric
    """
    gleak, gklt, gh, erev = params
    soma.g_leak = nstomho(gleak, somaarea)
    soma.gkltbar_LT_dth = nstomho(gklt, somaarea)
    soma.ghbar_IH_nmb = nstomho(gh, somaarea)
    soma.erev_leak = erev

    simulated_voltages = []

    for i in exp_currents:
        st.amp = i
        v_vec.resize(0)
        t_vec.resize(0)
        v_vec.record(soma(0.5)._ref_v)
        t_vec.record(h._ref_t)
        h.finitialize(-70)
        h.run()

        time_array = np.array(t_vec)
        voltage_array = np.array(v_vec)
        ss_mask = (time_array >= tmin) & (time_array <= tmax)
        simulated_voltages.append(np.mean(voltage_array[ss_mask]))

    simulated_voltages = np.array(simulated_voltages)
    ess = np.sum((exp_steady_state_voltages - simulated_voltages) ** 2)
    return ess

def lowpass_filter(data, cutoff=5000, fs=40000, order=2):
    """Apply a zero-phase Butterworth low-pass filter to the data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def run_unified_simulation(MNTB_class, param_dict, stim_amp=0.2, stim_delay=10, stim_dur=300,
                            v_init=-70, total_duration=510, dt=0.02, return_stim=False):
    """
    Run a NEURON simulation with standardized initialization, stimulation, and recording.

    Parameters:
        MNTB_class: constructor for your neuron model (e.g., MNTB)
        param_dict: dictionary with all required conductances and kinetic parameters
        stim_amp: amplitude of current injection (nA)
        stim_delay: onset of stimulus (ms)
        stim_dur: duration of stimulus (ms)
        v_init: initial membrane potential (mV)
        total_duration: simulation end time (ms)
        dt: time step (ms)
        return_stim: whether to return stimulus vector

    Returns:
        t: time vector (ms)
        v: voltage trace (mV)
        i_stim (optional): stimulus vector (nA)
    """

    # === Time and environment setup ===
    h.dt = 0.02
    h.steps_per_ms = int(1 / dt)
    h.celsius = 35

    # === Build cell from parameters ===
    cell = MNTB_class(
        gid=0,
        somaarea=param_dict['somaarea'],
        erev=param_dict['erev'],
        gleak=param_dict['gleak'],
        ena=param_dict['ena'],
        gna=param_dict['gna'],
        gh=param_dict['gh'],
        gka=param_dict['gka'],
        gklt=param_dict['gklt'],
        gkht=param_dict['gkht'],
        ek=param_dict['ek'],
        cam=param_dict.get('cam', 0),
        kam=param_dict.get('kam', 0),
        cbm=param_dict.get('cbm', 0),
        kbm=param_dict.get('kbm', 0),
        cah=param_dict.get('cah', 0),
        kah=param_dict.get('kah', 0),
        cbh=param_dict.get('cbh', 0),
        kbh=param_dict.get('kbh', 0),
    )

    # === Stimulus ===
    stim = h.IClamp(cell.soma(0.5))
    stim.delay = stim_delay
    stim.dur = stim_dur
    stim.amp = stim_amp

    # === Recording ===
    t_vec = h.Vector().record(h._ref_t)
    v_vec = h.Vector().record(cell.soma(0.5)._ref_v)
    i_vec = h.Vector().record(stim._ref_i)

    # === Initialization ===
    custom_init(v_init)

    # === Run simulation ===
    h.tstop = total_duration
    h.continuerun(total_duration)

    if return_stim:
        return np.array(t_vec), np.array(v_vec), np.array(i_vec)
    else:
        return np.array(t_vec), np.array(v_vec)

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if not 0 < normal_cutoff < 1:
        raise ValueError(f"⚠️ Invalid normalized cutoff: {normal_cutoff:.3f} (fs={fs}, cutoff={cutoff})")
    return butter(order, normal_cutoff, btype='low', analog=False)

def lowpass_filter(data, cutoff=2000, fs=50000, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def extract_features(trace, time, threspass=20):
    dt = time[1] - time[0]

    # Build spline for smooth derivative calculation
    spline = CubicSpline(time, trace, bc_type='not-a-knot', extrapolate=True)
    voltage = spline(time)
    dvdt = spline(time, nu=1)

    # === Resting potential (first 5 ms)
    rest = np.mean(voltage[:int(5 / dt)])

    # === Peak
    peak_idx = np.argmax(voltage)
    peak = voltage[peak_idx]

    # === Threshold detection: where dV/dt > 25 after 11 ms
    start_idx = int(11/ dt)
    try:
        rel_thresh_idx = np.where(dvdt[start_idx:] > threspass)[0][0]
        thresh_idx = start_idx + rel_thresh_idx
        threshold = voltage[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        return {
            'rest': rest, 'peak': peak, 'amp': np.nan,
            'threshold': np.nan, 'latency': np.nan,
            'width': np.nan, 'AHP': np.nan
        }

    amp = peak - threshold
    half_amp = threshold + 0.5 * amp

    # === Width at half-amplitude
    above_half = np.where(
        (voltage > half_amp) &
        (np.arange(len(voltage)) > thresh_idx) &
        (np.arange(len(voltage)) < peak_idx + int(5 / dt))
    )[0]

    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 1 else np.nan


    # === After-hyperpolarization (restricted to AP window)
    ahp_window_ms = (0.5, 5)  # AHP search: 0.5 to 10 ms after peak
    start_ahp_idx = peak_idx + int(ahp_window_ms[0] / dt)
    end_ahp_idx = peak_idx + int(ahp_window_ms[1] / dt)
    if start_ahp_idx < len(voltage):
        end_ahp_idx = min(end_ahp_idx, len(voltage))
        AHP = np.min(voltage[start_ahp_idx:end_ahp_idx])
    else:
        AHP = np.nan

    return {
        'rest': rest,
        'peak': peak,
        'amp': amp,
        'threshold': threshold,
        'latency': latency,
        'width': width,
        'AHP': AHP
    }

from scipy.interpolate import CubicSpline
import numpy as np

def extract_features_tent(trace, time, threspass=25, stim_delay=10.0, buffer=1.0,
                          max_latency=300.0, ahp_window=(0.5, 10.0), verbose=False):
    dt = time[1] - time[0]
    spline = CubicSpline(time, trace, bc_type='not-a-knot', extrapolate=True)

    voltage = spline(time)
    dvdt = spline(time, nu=1)

    # === Resting potential before stimulus
    rest = np.mean(voltage[:int(stim_delay / dt)])

    # === Threshold detection window
    start_idx = int((stim_delay + buffer) / dt)
    end_idx = int((stim_delay + max_latency) / dt)

    try:
        rel_thresh_idx = np.where(dvdt[start_idx:end_idx] > threspass)[0][0]
        thresh_idx = start_idx + rel_thresh_idx
        threshold = voltage[thresh_idx]
        latency = time[thresh_idx]
    except IndexError:
        if verbose:
            print(f"❌ No AP detected (dV/dt never > {threspass} between {stim_delay+buffer}-{stim_delay+max_latency} ms)")
        return {
            'rest': rest, 'peak': np.nan, 'amp': np.nan,
            'threshold': np.nan, 'latency': np.nan,
            'width': np.nan, 'AHP': np.nan
        }

    # === Peak detection after threshold
    peak_idx = np.argmax(voltage[thresh_idx:]) + thresh_idx
    peak = voltage[peak_idx]
    amp = peak - threshold
    half_amp = threshold + 0.5 * amp

    # === Width at half-amplitude
    above_half = np.where(
        (voltage > half_amp) &
        (np.arange(len(voltage)) > thresh_idx) &
        (np.arange(len(voltage)) < peak_idx + int(5 / dt))
    )[0]
    width = (above_half[-1] - above_half[0]) * dt if len(above_half) > 1 else np.nan

    # === AHP (within limited window after peak)
    start_ahp_idx = peak_idx + int(ahp_window[0] / dt)
    end_ahp_idx = min(peak_idx + int(ahp_window[1] / dt), len(voltage))
    AHP = np.min(voltage[start_ahp_idx:end_ahp_idx]) if end_ahp_idx > start_ahp_idx else np.nan

    if verbose:
        print(f"✅ AP detected | Latency: {latency:.2f} ms | Threshold: {threshold:.2f} mV | Peak: {peak:.2f} mV | AHP: {AHP:.2f} mV")

    return {
        'rest': rest,
        'peak': peak,
        'amp': amp,
        'threshold': threshold,
        'latency': latency,
        'width': width,
        'AHP': AHP
    }

