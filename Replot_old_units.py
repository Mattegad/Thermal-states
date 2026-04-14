from __future__ import annotations

"""
Replot utility for data saved by polariton_microcavity_homodyne_simulation.py

Usage
-----
python replot_results.py polariton_homodyne_results.npz

This script:
- reloads the saved .npz results file
- restores metadata (including complex values encoded in JSON)
- prints a summary of what is available
- replots time traces, phase-space clouds, spectra, LO phase sweep diagnostics,
  and optional integrated-band quantities
- can also save the figures to disk

You can adapt the defaults in the CONFIG section at the bottom.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import welch
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# -----------------------------------------------------------------------------
# JSON safe
# -----------------------------------------------------------------------------


def json_safe(obj):
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------------

def restore_complex(obj: Any) -> Any:
    """Recursively restore complex values encoded as tagged dicts."""
    if isinstance(obj, dict):
        if obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])
        return {k: restore_complex(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restore_complex(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def load_results_npz(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file: {path}")

    data = np.load(path, allow_pickle=True)
    files = list(data.files)

    metadata: Dict[str, Any] = {}
    if "metadata_json" in files:
        raw = data["metadata_json"]
        if isinstance(raw, np.ndarray):
            raw = raw.item()
        metadata = restore_complex(json.loads(str(raw)))

    results: Dict[str, np.ndarray] = {}
    for key in files:
        if key == "metadata_json":
            continue
        results[key] = data[key]

    return metadata, results


# -----------------------------------------------------------------------------
# Math / diagnostics helpers
# -----------------------------------------------------------------------------

def complex_to_quadratures(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sqrt(2.0) * np.real(z)
    p = np.sqrt(2.0) * np.imag(z)
    return x, p


def quadrature_projection(z: np.ndarray, theta: float) -> np.ndarray:
    return np.sqrt(2.0) * np.real(np.exp(-1j * theta) * z)


def estimate_quadrature_variances_from_complex(z: np.ndarray) -> Dict[str, float]:
    x, p = complex_to_quadratures(z)
    cov = np.cov(x, p, ddof=0)
    return {
        "mean_x": float(np.mean(x)),
        "mean_p": float(np.mean(p)),
        "var_x": float(np.var(x)),
        "var_p": float(np.var(p)),
        "cov_xp": float(cov[0, 1]),
    }


def infer_sample_rate_hz(results: Dict[str, np.ndarray]) -> float:
    if "fs_store_hz" in results:
        val = results["fs_store_hz"]
        if np.ndim(val) == 0:
            return float(val)
        return float(np.ravel(val)[0])
    if "t_s" in results and len(results["t_s"]) > 1:
        dt = float(results["t_s"][1] - results["t_s"][0])
        return 1.0 / dt
    raise ValueError("Could not infer sample rate from the results file.")


def compute_psd(
    x: np.ndarray,
    fs_hz: float,
    nperseg: int = 2**14,
    window: str = "hann",
    detrend: str = "constant",
    average: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    if SCIPY_AVAILABLE:
        f, pxx = welch(
            x,
            fs=fs_hz,
            window=window,
            nperseg=min(nperseg, len(x)),
            detrend=detrend,
            return_onesided=True,
            scaling="density",
            average=average,
        )
        return f, pxx

    n = len(x)
    if n < 2:
        raise ValueError("Need at least two points to compute a PSD.")
    win = np.hanning(n)
    xw = (x - np.mean(x)) * win
    norm = fs_hz * np.sum(win**2)
    Xf = np.fft.rfft(xw)
    pxx = (np.abs(Xf) ** 2) / norm
    f = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    return f, pxx


def rbw_average_psd(freqs_hz: np.ndarray, psd: np.ndarray, rbw_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    if rbw_hz <= 0 or len(freqs_hz) < 2:
        return freqs_hz, psd
    df = freqs_hz[1] - freqs_hz[0]
    bins = max(1, int(round(rbw_hz / df)))
    if bins <= 1:
        return freqs_hz, psd
    n = (len(psd) // bins) * bins
    if n == 0:
        return freqs_hz, psd
    return freqs_hz[:n].reshape(-1, bins).mean(axis=1), psd[:n].reshape(-1, bins).mean(axis=1)


def integrated_band_power(freqs_hz: np.ndarray, psd: np.ndarray, fmin_hz: Optional[float], fmax_hz: Optional[float]) -> float:
    mask = np.ones_like(freqs_hz, dtype=bool)
    if fmin_hz is not None:
        mask &= freqs_hz >= fmin_hz
    if fmax_hz is not None:
        mask &= freqs_hz <= fmax_hz
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(psd[mask], freqs_hz[mask]))


def psd_value_at(freqs_hz: np.ndarray, psd: np.ndarray, f0_hz: float) -> float:
    idx = int(np.argmin(np.abs(freqs_hz - f0_hz)))
    return float(psd[idx])

# -----------------------------------------------------------------------------
# Detection-mode helpers
# -----------------------------------------------------------------------------

def get_detection_mode(metadata: Dict[str, Any]) -> Optional[str]:
    detection = metadata.get("detection", {})
    if isinstance(detection, dict):
        return str(detection.get("mode", "unknown"))
    return "unknown"


def get_lo_phase_from_metadata(metadata: Dict[str, Any], default: float = np.pi / 2) -> float:
    detection = metadata.get("detection", {})
    if isinstance(detection, dict):
        return float(detection.get("lo_phase_rad", default))
    return default


def get_primary_detected_channels(
    results: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Returns:
        det_signal, meas_signal, label
    """
    mode = get_detection_mode(metadata)

    if mode == "homodyne":
        return results.get("i_det_t"), results.get("i_meas_t"), "homodyne"

    if mode == "balanced_sum":
        det = results.get("i_plus_det_t", results.get("i_det_t"))
        meas = results.get("i_plus_meas_t", results.get("i_meas_t"))
        return det, meas, "i+"

    if mode == "balanced_diff":
        det = results.get("i_minus_det_t", results.get("i_det_t"))
        meas = results.get("i_minus_meas_t", results.get("i_meas_t"))
        return det, meas, "i-"

    return results.get("i_det_t"), results.get("i_meas_t"), "detected signal"


# -----------------------------------------------------------------------------
# Derived channels
# -----------------------------------------------------------------------------

def build_missing_channels_from_output(
    results: Dict[str, np.ndarray], 
    metadata: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Reconstruct useful channels if some were not saved."""
    out = dict(results)

    lo_phase_rad = get_lo_phase_from_metadata(metadata)

    if "s_out_t" in out:
        z = out["s_out_t"]
        if "x_out" not in out or "p_out" not in out:
            x_out, p_out = complex_to_quadratures(z)
            out.setdefault("x_out", x_out)
            out.setdefault("p_out", p_out)

    if "F_t" in out:
        z = out["F_t"]
        if "x_in" not in out or "p_in" not in out:
            x_in, p_in = complex_to_quadratures(z)
            out.setdefault("x_in", x_in)
            out.setdefault("p_in", p_in)

    if "psi_t" in out:
        z = out["psi_t"]
        if "x_cav" not in out or "p_cav" not in out:
            x_cav, p_cav = complex_to_quadratures(z)
            out.setdefault("x_cav", x_cav)
            out.setdefault("p_cav", p_cav)

    if "i_det_t" not in out and "s_out_t" in out:
        out["i_det_t"] = quadrature_projection(out["s_out_t"], lo_phase_rad)

    if "i_meas_t" not in out and "i_det_t" in out:
        out["i_meas_t"] = out["i_det_t"].copy()

    if "freqs_det_hz" not in out and "i_det_t" in out:
        fs = infer_sample_rate_hz(out)
        f, p = compute_psd(out["i_det_t"], fs_hz=fs)
        out["freqs_det_hz"] = f
        out["psd_det"] = p

    if "freqs_meas_hz" not in out and "i_meas_t" in out:
        fs = infer_sample_rate_hz(out)
        f, p = compute_psd(out["i_meas_t"], fs_hz=fs)
        out["freqs_meas_hz"] = f
        out["psd_meas"] = p

    return out


# -----------------------------------------------------------------------------
# Pretty printing
# -----------------------------------------------------------------------------

def print_metadata_summary(metadata: Dict[str, Any]) -> None:
    if not metadata:
        print("No metadata found in file.")
        return

    print("\n=== Metadata summary ===")
    for section, content in metadata.items():
        print(f"[{section}]")
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {content}")


def print_results_summary(results: Dict[str, np.ndarray]) -> None:
    print("\n=== Saved arrays ===")
    for k, v in results.items():
        shape = getattr(v, "shape", None)
        dtype = getattr(v, "dtype", None)
        print(f"{k:>15s} : shape={shape}, dtype={dtype}")

    for key in ["F_t", "psi_t", "s_out_t"]:
        if key in results:
            stats = estimate_quadrature_variances_from_complex(results[key])
            print(f"\n=== Quadrature stats for {key} ===")
            for name, value in stats.items():
                print(f"  {name:>10s} = {value:.6g}")


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_time_traces(
    results: Dict[str, np.ndarray], 
    metadata : Dict[str, Any],
    max_points: int = 6000,
) -> plt.Figure:
    t = results["t_s"]
    n = len(t)
    step = max(1, n // max_points)
    sl = slice(None, None, step)

    mode = get_detection_mode(metadata)

    if mode == "homodyne":
        nrows = 5
    else:
        nrows = 6
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 3 * nrows), sharex=True)
    row = 0

    if "amp_noise" in results or "phase_noise" in results:
        if "amp_noise" in results:
            axes[row].plot(t[sl] * 1e6, results["amp_noise"][sl], label="Amplitude noise")
        if "phase_noise" in results:
            axes[row].plot(t[sl] * 1e6, results["phase_noise"][sl], label="Phase noise")
        axes[row].legend()
    axes[row].set_ylabel("Noise")
    axes[row].grid(True, alpha=0.3)
    row += 1

    if "F_t" in results:
        axes[row].plot(t[sl] * 1e6, np.real(results["F_t"][sl]), label="Re F(t)")
        axes[row].plot(t[sl] * 1e6, np.imag(results["F_t"][sl]), label="Im F(t)")
        axes[row].legend()
    axes[row].set_ylabel("Drive")
    axes[row].grid(True, alpha=0.3)
    row += 1

    if "psi_t" in results:
        axes[row].plot(t[sl] * 1e6, np.real(results["psi_t"][sl]), label="Re ψ(t)")
        axes[row].plot(t[sl] * 1e6, np.imag(results["psi_t"][sl]), label="Im ψ(t)")
        axes[row].legend()
    axes[row].set_ylabel("Intracavity")
    axes[row].grid(True, alpha=0.3)
    row += 1

    if "s_out_t" in results:
        axes[row].plot(t[sl] * 1e6, np.real(results["s_out_t"][sl]), label="Re s_out(t)")
        axes[row].plot(t[sl] * 1e6, np.imag(results["s_out_t"][sl]), label="Im s_out(t)")
        axes[row].legend()
    axes[row].set_ylabel("Output")
    axes[row].grid(True, alpha=0.3)
    row += 1

    if mode == "homodyne":
        if "i_det_t" in results:
            axes[row].plot(t[sl] * 1e6, results["i_det_t"][sl], label="i_det(t)")
        if "i_meas_t" in results and "i_meas_t" in results:
            axes[row].plot(t[sl] * 1e6, results["i_meas_t"][sl], label="i_meas(t)", alpha=0.7)
        axes[row].legend()
        axes[row].set_ylabel("Photocurrent")
        axes[row].set_xlabel("Time (µs)")
        axes[row].grid(True, alpha=0.3)
    else:
        if "i1_meas_t" in results:
            axes[row].plot(t[sl] * 1e6, results["i1_meas_t"][sl], label="i1_meas(t)")
        if "i2_meas_t" in results:
            axes[row].plot(t[sl] * 1e6,         results["i2_meas_t"][sl], label="i2_meas(t)", alpha=0.7)
        axes[row].legend()
        axes[row].set_ylabel("Diodes currents")
        axes[row].set_xlabel("Time (µs)")
        axes[row].grid(True, alpha=0.3)
        row += 1

    if "i_plus_meas_t" in results:
        axes[row].plot(t[sl] * 1e6, results["i_plus_meas_t"][sl], label="i+ meas(t)")
    if "i_minus_meas_t" in results:
        axes[row].plot(t[sl] * 1e6, results["i_minus_meas_t"][sl], label="i- meas(t)", alpha=0.7)
    axes[row].legend()
    axes[row].set_ylabel("Balanced meas.")
    axes[row].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (µs)")
    fig.tight_layout()
    return fig


def plot_phase_space(results: Dict[str, np.ndarray], max_points: int = 50000) -> plt.Figure:
    available = []
    if "x_in" in results and "p_in" in results:
        available.append((results["x_in"], results["p_in"], "Input drive quadratures", "X_in", "P_in"))
    if "x_cav" in results and "p_cav" in results:
        available.append((results["x_cav"], results["p_cav"], "Intracavity quadratures", "X_cav", "P_cav"))
    if "x_out" in results and "p_out" in results:
        available.append((results["x_out"], results["p_out"], "Output field quadratures", "X_out", "P_out"))

    nplots = max(1, len(available))
    fig, axes = plt.subplots(1, nplots, figsize=(6 * nplots, 5))
    if nplots == 1:
        axes = [axes]

    for ax, (x, p, title, xlabel, ylabel) in zip(axes, available):
        step = max(1, len(x) // max_points)
        sl = slice(None, None, step)
        ax.scatter(x[sl], p[sl], s=2, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_quadratures_vs_time(results: Dict[str, np.ndarray], max_points: int = 6000) -> plt.Figure:
    t = results["t_s"]
    n = len(t)
    step = max(1, n // max_points)
    sl = slice(None, None, step)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    if "x_in" in results and "p_in" in results:
        axes[0].plot(t[sl] * 1e6, results["x_in"][sl], label="X_in")
        axes[0].plot(t[sl] * 1e6, results["p_in"][sl], label="P_in")
        axes[0].legend()
    axes[0].set_ylabel("Input quadratures")
    axes[0].grid(True, alpha=0.3)

    if "x_cav" in results and "p_cav" in results:
        axes[1].plot(t[sl] * 1e6, results["x_cav"][sl], label="X_cav")
        axes[1].plot(t[sl] * 1e6, results["p_cav"][sl], label="P_cav")
        axes[1].legend()
    axes[1].set_ylabel("Cavity quadratures")
    axes[1].grid(True, alpha=0.3)

    if "x_out" in results and "p_out" in results:
        axes[2].plot(t[sl] * 1e6, results["x_out"][sl], label="X_out")
        axes[2].plot(t[sl] * 1e6, results["p_out"][sl], label="P_out")
        axes[2].legend()
    axes[2].set_ylabel("Output quadratures")
    axes[2].set_xlabel("Time (µs)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spectra(
    results: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    rbw_hz: float = 0.0,
    fmin_hz: Optional[float] = None,
    fmax_hz: Optional[float] = None,
    loglog: bool = False,
) -> plt.Figure:
    f1 = results["freqs_det_hz"]
    p1 = results["psd_det"]
    f2 = results["freqs_meas_hz"]
    p2 = results["psd_meas"]

    if rbw_hz > 0:
        f1, p1 = rbw_average_psd(f1, p1, rbw_hz)
        f2, p2 = rbw_average_psd(f2, p2, rbw_hz)

    mask1 = np.ones_like(f1, dtype=bool)
    mask2 = np.ones_like(f2, dtype=bool)
    if fmin_hz is not None:
        mask1 &= f1 >= fmin_hz
        mask2 &= f2 >= fmin_hz
    if fmax_hz is not None:
        mask1 &= f1 <= fmax_hz
        mask2 &= f2 <= fmax_hz

    _, _, label = get_primary_detected_channels(results, metadata)

    fig = plt.figure(figsize=(10, 6))
    if loglog:
        plt.loglog(f1[mask1], p1[mask1], label=f"PSD deterministic ({label})")
        plt.loglog(f2[mask2], p2[mask2], label=f"PSD measured ({label})")
    else:
        plt.semilogy(f1[mask1], p1[mask1], label=f"PSD deterministic ({label})")
        plt.semilogy(f2[mask2], p2[mask2], label=f"PSD measured ({label})")
    plt.xlabel("Analysis frequency (Hz)")
    plt.ylabel("PSD (current units²/Hz)")
    plt.title("Spectrum analyzer trace")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_band_power(results: Dict[str, np.ndarray], use_measured: bool = True) -> plt.Figure:
    f = results["freqs_meas_hz"] if use_measured else results["freqs_det_hz"]
    p = results["psd_meas"] if use_measured else results["psd_det"]
    df = np.diff(f)
    if len(df) == 0:
        raise ValueError("Not enough frequency points.")
    cum = np.zeros_like(f)
    cum[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * df)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(f, cum)
    plt.xlabel("Upper integration frequency (Hz)")
    plt.ylabel("Integrated noise power")
    plt.title("Cumulative integrated PSD")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_all_balanced_psds(
    results: Dict[str, np.ndarray],
    rbw_hz: float = 0.0,
    fmin_hz: Optional[float] = None,
    fmax_hz: Optional[float] = None,
    loglog: bool = False,
) -> Optional[Dict[str, plt.Figure]]:
    """
    Plot i1, i2, i+, i- measured PSD's if available.
    Useful only for balanced detection
    """
    needed = ["i1_meas_t", "i2_meas_t", "i_plus_meas_t", "i_minus_meas_t"]
    if not all(k in results for k in needed):
        return None
    
    fs = infer_sample_rate_hz(results)

    curves = []
    for key, label in [("i1_meas_t", "i1"), ("i2_meas_t", "i2"), ("i_plus_meas_t", "i+"), ("i_minus_meas_t", "i-")]:
        f, p = compute_psd(results[key], fs_hz=fs)
        if rbw_hz > 0:
            f, p = rbw_average_psd(f, p, rbw_hz)
        curves.append((f, p, label))
    
    fig = plt.figure(figsize=(10, 6))
    for f, p, label in curves:
        mask = np.ones_like(f, dtype=bool)
        if fmin_hz is not None:
            mask &= f >= fmin_hz
        if fmax_hz is not None:
            mask &= f <= fmax_hz
        plt.semilogy(f[mask], p[mask], label=label)
    
    plt.xlabel("Analysis frequency (Hz)")
    plt.ylabel("PSD (current units²/Hz)")
    plt.title("PSD of balanced detection channels")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fig.tight_layout()
    return fig

    
def plot_lo_phase_sweep_from_saved_signal(
    results: Dict[str, np.ndarray],
    n_phases: int = 41,
    use_psd_at_hz: Optional[float] = None,
    band: Optional[Tuple[Optional[float], Optional[float]]] = None,
    nperseg: int = 2**14,
) -> plt.Figure:
    if "s_out_t" not in results:
        raise ValueError("Need s_out_t to reconstruct a LO-phase sweep.")

    z = results["s_out_t"]
    fs = infer_sample_rate_hz(results)
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases)
    values = []

    for theta in phases:
        i_theta = quadrature_projection(z, theta)
        if use_psd_at_hz is None and band is None:
            values.append(float(np.var(i_theta)))
        else:
            f, p = compute_psd(i_theta, fs_hz=fs, nperseg=nperseg)
            if use_psd_at_hz is not None:
                values.append(psd_value_at(f, p, use_psd_at_hz))
            else:
                fmin_hz, fmax_hz = band if band is not None else (None, None)
                values.append(integrated_band_power(f, p, fmin_hz, fmax_hz))

    fig = plt.figure(figsize=(8, 5))
    plt.plot(phases, values, marker="o")
    plt.xlabel("LO phase θ (rad)")
    if use_psd_at_hz is None and band is None:
        plt.ylabel("Var[homodyne current]")
        plt.title("LO phase sweep from saved output field")
    elif use_psd_at_hz is not None:
        plt.ylabel(f"PSD at {use_psd_at_hz:.3g} Hz")
        plt.title("LO phase sweep at fixed analysis frequency")
    else:
        fmin_hz, fmax_hz = band if band is not None else (None, None)
        plt.ylabel("Integrated PSD")
        plt.title(f"LO phase sweep integrated over {fmin_hz:.3g}–{fmax_hz:.3g} Hz")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Saving figures
# -----------------------------------------------------------------------------

def save_figure(fig: plt.Figure, outdir: Path, name: str, dpi: int = 180) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / name, dpi=dpi, bbox_inches="tight")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    # -----------------------
    # CONFIG
    # -----------------------
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("polariton_homodyne_results.npz")
    save_figures = True
    output_dir = Path("Plots")

    # spectrum display choices
    fmin_hz = 1e3
    fmax_hz = 1.2e8
    rbw_hz = 3e4
    loglog = False

    # LO phase sweep replot choices
    lo_sweep_phases = 41
    lo_sweep_use_psd_at_hz = 1e6      # set to None to disable fixed-frequency PSD mode
    lo_sweep_band = None              # example: (5e5, 2e6). If not None, used instead of var().

    # -----------------------
    # Load
    # -----------------------
    metadata, results = load_results_npz(input_path)
    results = build_missing_channels_from_output(results)

    meta_path = output_dir / "metadata_replot.json"

    with open(meta_path, "w") as f:
        json.dump(json_safe(metadata), f, indent=2)

    print(f"Metadata saved to {meta_path}")
    print(f"Loaded: {input_path.resolve()}")

    print_metadata_summary(metadata)
    print_results_summary(results)

    mode = get_detection_mode(metadata)
    print(f"\nInferred detection mode: {mode}")

    # useful scalar summaries
    #if "freqs_meas_hz" in results and "psd_meas" in results:
        #band_power = integrated_band_power(results["freqs_meas_hz"], results["psd_meas"], fmin_hz, fmax_hz)
        #print(f"\nIntegrated measured PSD over [{fmin_hz:.3g}, {fmax_hz:.3g}] Hz = {band_power:.6g}")
        #print(f"Measured PSD at 1 MHz = {psd_value_at(results['freqs_meas_hz'], results['psd_meas'], 1e6):.6g}")


    # -----------------------
    # Replots
    # -----------------------
    figs: Dict[str, plt.Figure] = {}
    figs["01_time_traces.png"] = plot_time_traces(results)
    figs["02_quadratures_vs_time.png"] = plot_quadratures_vs_time(results)
    figs["03_phase_space.png"] = plot_phase_space(results)
    figs["04_spectra.png"] = plot_spectra(results, metadata, rbw_hz=rbw_hz, fmin_hz=fmin_hz, fmax_hz=fmax_hz, loglog=loglog)
    figs["05_cumulative_band_power.png"] = plot_cumulative_band_power(results, use_measured=True)

    balanced_psd_fig = plot_all_balanced_psds(results, rbw_hz=rbw_hz, fmin_hz=fmin_hz, fmax_hz=fmax_hz)

    if balanced_psd_fig is not None:
        figs["06_balanced_psds.png"] = balanced_psd_fig
        next_index = 7
    else:
        next_index = 6

    if "s_out_t" in results:
        figs[f"0{next_index:02d}_lo_phase_sweep_var.png"] = plot_lo_phase_sweep_from_saved_signal(
            results,
            n_phases=lo_sweep_phases,
            use_psd_at_hz=None,
            band=None,
        )
        next_index += 1

        if lo_sweep_use_psd_at_hz is not None:
            figs[f"0{next_index:02d}_lo_phase_sweep_psd_fixed_freq.png"] = plot_lo_phase_sweep_from_saved_signal(
                results,
                n_phases=lo_sweep_phases,
                use_psd_at_hz=lo_sweep_use_psd_at_hz,
                band=None,
            )
            next_index += 1

        if lo_sweep_band is not None:
            figs[f"0{next_index:02d}_lo_phase_sweep_psd_band.png"] = plot_lo_phase_sweep_from_saved_signal(
                results,
                n_phases=lo_sweep_phases,
                use_psd_at_hz=None,
                band=lo_sweep_band,
            )

    if save_figures:
        for name, fig in figs.items():
            save_figure(fig, output_dir, name)
        print(f"\nSaved figures to: {output_dir.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
