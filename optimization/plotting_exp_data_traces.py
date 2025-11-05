#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
from matplotlib import rcParams
from pathlib import Path

rcParams['pdf.fonttype'] = 42   # TrueType
rcParams['ps.fonttype'] = 42    # For EPS too, if needed

# ---------- helpers ----------
def parse_pair_floats(s):
    # parse "a,b" into (a, b)
    try:
        a, b = s.split(",")
        return (float(a.strip()), float(b.strip()))
    except Exception:
        raise argparse.ArgumentTypeError(f"Expected 'a,b' float pair, got '{s}'")

def load_voltage_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df is None or df.empty:
        raise ValueError(f"Empty DataFrame: {csv_path}")
    return df

def rename_sweeps_if_needed(df: pd.DataFrame, filename_stem: str,
                            start_amp=-0.1, step=0.02):
    """Your original logic: if 2nd column starts with 'Sweep', rename columns to amps."""
    if df.shape[1] < 2:
        return df
    if str(df.columns[1]).startswith("Sweep"):
        n_sweeps = df.shape[1] - 1
        amps = np.round(np.arange(start_amp, start_amp + step * n_sweeps, step), 3)
        amp_labels = [f"{amp} nA" for amp in amps]
        df.columns = [df.columns[0]] + amp_labels
    return df

def detect_rheobase_col(df: pd.DataFrame, time_limit_ms=200, spike_threshold=-20):
    """Return the column name of the first trace that crosses spike_threshold before time_limit."""
    time = df.iloc[:, 0]
    time_mask = time <= time_limit_ms
    for col in df.columns[1:]:
        trace = pd.to_numeric(df[col], errors="coerce")
        if (trace[time_mask] > spike_threshold).any():
            return col
    return None

def plot_voltage_traces(df_voltage: pd.DataFrame,
                        filename_stem: str,
                        xlim=(0, 400),
                        ylim=(-120, 40),
                        save_fig=False,
                        dpi=300,
                        output_dir: Path = None):
    """
    Plots voltage traces and stops at rheobase (first spiking trace).
    Saves PNG/PDF if save_fig=True.
    """
    if df_voltage is None or df_voltage.empty:
        print("DataFrame is empty or None. Nothing to plot.")
        return

    # Make all non-time columns numeric
    for col in df_voltage.columns[1:]:
        df_voltage[col] = pd.to_numeric(df_voltage[col], errors='coerce')

    time = df_voltage.iloc[:, 0]

    # color theme based on filename
    if "iMNTB" in filename_stem:
        colors = ['black', 'gray']
        rheo_color = 'black'
        extra_color = 'gray'
    else:
        colors = ['red', 'pink']
        rheo_color = 'red'
        extra_color = 'pink'

    # Find rheobase
    rheobase_col = detect_rheobase_col(df_voltage, time_limit_ms=200, spike_threshold=-20)

    # Build sweep list and ensure we include last trace + a 200 pA example if present
    sweep_cols = df_voltage.columns[1:].tolist()
    last_col = df_voltage.columns[-1]
    # Choose a “+200 pA” column by index if present (your note: 16 for 20 pA steps, 31 for 10 pA steps)
    # We’ll try 16 first, then 31, else skip gracefully.
    col_200pA = None
    for idx in (16, 31):
        if df_voltage.shape[1] > idx:
            col_200pA = df_voltage.columns[idx]
            break

    if rheobase_col and rheobase_col not in sweep_cols:
        sweep_cols.append(rheobase_col)
    if last_col not in sweep_cols:
        sweep_cols.append(last_col)

    # Sort by numeric amplitude if column names look like "<value> nA"
    def sort_key(c):
        try:
            return float(str(c).split()[0])
        except Exception:
            return np.inf
    sweep_cols_sorted = sorted(sweep_cols, key=sort_key)

    # Plot
    plt.figure(figsize=(10, 10))
    for i, col in enumerate(sweep_cols_sorted):
        trace = df_voltage[col]
        color = rheo_color if (rheobase_col is not None and col == rheobase_col) else colors[i % 2]
        plt.plot(time, trace, color=color, linewidth=1)
        if rheobase_col is not None and col == rheobase_col:
            break  # stop after rheobase

    if col_200pA and col_200pA != rheobase_col and col_200pA in df_voltage.columns:
        trace = df_voltage[col_200pA]
        plt.plot(time, trace, color=extra_color, linewidth=1, linestyle=':')

    # Cosmetics
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(False)
    plt.tight_layout()
    plt.axis('off')

    # Scale bars
    x_scale = 50  # ms
    y_scale = 20  # mV
    xlim_applied = xlim
    ylim_applied = ylim
    x_start = xlim_applied[1] - 2.51 * x_scale
    y_start = ylim_applied[0] + 0.03 * (ylim_applied[1] - ylim_applied[0])
    plt.hlines(y=y_start, xmin=x_start, xmax=x_start + x_scale, linewidth=2, color='black')
    plt.text(x_start + x_scale / 2, y_start - 0.03 * (ylim[1] - ylim[0]), f"{x_scale} ms",
             ha='center', va='top', fontsize=14)
    plt.vlines(x=x_start, ymin=y_start, ymax=y_start + y_scale, linewidth=2, color='black')
    plt.text(x_start - 0.01 * (xlim[1] - xlim[0]), y_start + y_scale / 2, f"{y_scale} mV",
             ha='right', va='center', fontsize=14, rotation=90)

    # Save if requested
    if save_fig:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / ".." / "figures" / "exp_data_traces"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_base = output_dir / filename_stem
        plt.savefig(f"{out_base}.png", dpi=dpi, bbox_inches='tight')
        plt.savefig(f"{out_base}.pdf", dpi=dpi, bbox_inches='tight')
        print(f"✅ Figures saved to:\n{out_base}.png\n{out_base}.pdf")

    plt.show()

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Plot voltage sweeps and stop at rheobase.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to a single CSV file.")
    src.add_argument("--data_dir", help="Directory with CSV files. Use with --filename or --batch.")
    parser.add_argument("--filename", help="CSV filename inside --data_dir (ignored if --csv is used).")
    parser.add_argument("--batch", action="store_true", help="Process all *.csv in --data_dir (ignores --filename).")

    parser.add_argument("--xlim", type=parse_pair_floats, default="0,400", help="X limits as 'min,max' (ms).")
    parser.add_argument("--ylim", type=parse_pair_floats, default="-120,40", help="Y limits as 'min,max' (mV).")
    parser.add_argument("--save_fig", action="store_true", help="Save PNG/PDF to ../figures/exp_data_traces by default.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving.")
    parser.add_argument("--output_dir", default=None, help="Optional custom output directory.")
    parser.add_argument("--start_amp", type=float, default=-0.1, help="Starting amp for relabeling Sweeps.")
    parser.add_argument("--step_amp", type=float, default=0.02, help="Step amp for relabeling Sweeps.")
    args = parser.parse_args()

    # Resolve script_dir for defaults
    script_dir = Path(__file__).resolve().parent

    # Output directory handling
    out_dir = Path(args.output_dir) if args.output_dir else None

    def process_one(csv_path: Path):
        df = load_voltage_csv(csv_path)
        filename_stem = csv_path.stem
        df = rename_sweeps_if_needed(df, filename_stem, start_amp=args.start_amp, step=args.step_amp)
        plot_voltage_traces(
            df,
            filename_stem=filename_stem,
            xlim=args.xlim,
            ylim=args.ylim,
            save_fig=args.save_fig,
            dpi=args.dpi,
            output_dir=out_dir
        )

    if args.csv:
        process_one(Path(args.csv).expanduser())
    else:
        data_dir = Path(args.data_dir).expanduser()
        if args.batch:
            for csv_path in sorted(data_dir.glob("*.csv")):
                print(f"--- Processing {csv_path.name} ---")
                process_one(csv_path)
        else:
            if not args.filename:
                raise SystemExit("When using --data_dir without --batch, you must pass --filename.")
            process_one((data_dir / args.filename).expanduser())

if __name__ == "__main__":
    main()
