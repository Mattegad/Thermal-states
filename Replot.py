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
        return {"__complex__": True, "real": obj.real, "imag": obj.imag}
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


def infer_sample_rate_mhz(results: Dict[str, np.ndarray]) -> float:
    if "fs_store_mhz" in results:
        val = results["fs_store_mhz"]
        if np.ndim(val) == 0:
            return float(val)
        return float(np.ravel(val)[0])
    if "t_ps" in results and len(results["t_ps"]) > 1:
        dt = float(results["t_ps"][1] - results["t_ps"][0])
        return 1.0 / dt
    raise ValueError("Could not infer sample rate from the results file.")


def compute_psd(
    x: np.ndarray,
    fs_mhz: float,
    nperseg: int = 2**13,
    window: str = "hann",
    detrend: str = "constant",
    average: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    if SCIPY_AVAILABLE:
        f, pxx = welch(
            x,
            fs=fs_mhz,
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
    norm = fs_mhz * np.sum(win**2)
    Xf = np.fft.rfft(xw)
    pxx = (np.abs(Xf) ** 2) / norm
    f = np.fft.rfftfreq(n, d=1.0 / fs_mhz)
    return f, pxx


def rbw_average_psd(freqs_mhz: np.ndarray, psd: np.ndarray, rbw_mhz: float) -> Tuple[np.ndarray, np.ndarray]:
    if rbw_mhz <= 0 or len(freqs_mhz) < 2:
        return freqs_mhz, psd
    df = freqs_mhz[1] - freqs_mhz[0]
    bins = max(1, int(round(rbw_mhz / df)))
    if bins <= 1:
        return freqs_mhz, psd
    n = (len(psd) // bins) * bins
    if n == 0:
        return freqs_mhz, psd
    return freqs_mhz[:n].reshape(-1, bins).mean(axis=1), psd[:n].reshape(-1, bins).mean(axis=1)


def integrated_band_power(freqs_mhz: np.ndarray, psd: np.ndarray, fmin_mhz: Optional[float], fmax_mhz: Optional[float]) -> float:
    mask = np.ones_like(freqs_mhz, dtype=bool)
    if fmin_mhz is not None:
        mask &= freqs_mhz >= fmin_mhz
    if fmax_mhz is not None:
        mask &= freqs_mhz <= fmax_mhz
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(psd[mask], freqs_mhz[mask]))


def psd_value_at(freqs_mhz: np.ndarray, psd: np.ndarray, f0_mhz: float) -> float:
    idx = int(np.argmin(np.abs(freqs_mhz - f0_mhz)))
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

    if "freqs_det_mhz" not in out and "i_det_t" in out:
        fs = infer_sample_rate_mhz(out)
        f, p = compute_psd(out["i_det_t"], fs_mhz=fs)
        out["freqs_det_mhz"] = f
        out["psd_det"] = p

    if "freqs_meas_mhz" not in out and "i_meas_t" in out:
        fs = infer_sample_rate_mhz(out)
        f, p = compute_psd(out["i_meas_t"], fs_mhz=fs)
        out["freqs_meas_mhz"] = f
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
    metadata: Dict[str, Any],
    max_points: int = 6000,
) -> plt.Figure:
    t = results["t_ps"]
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
        if "i_meas_t" in results:
            axes[row].plot(t[sl] * 1e6, results["i_meas_t"][sl], label="i_meas(t)", alpha=0.7)
        axes[row].legend()
        axes[row].set_ylabel("Photocurrent")
        axes[row].set_xlabel(f"Time ($\mu$s)")
        axes[row].grid(True, alpha=0.3)
    else:
        if "i1_meas_t" in results:
            axes[row].plot(t[sl] * 1e6, results["i1_meas_t"][sl], label="i1_meas(t)")
        if "i2_meas_t" in results:
            axes[row].plot(t[sl] * 1e6, results["i2_meas_t"][sl], label="i2_meas(t)", alpha=0.7)
        axes[row].legend()
        axes[row].set_ylabel("Diodes currents")
        axes[row].set_xlabel(r"Time ($\mu$s)")
        axes[row].grid(True, alpha=0.3)
        row += 1

    if "i_plus_meas_t" in results:
        axes[row].plot(t[sl] * 1e6, results["i_plus_meas_t"][sl], label="i+ meas(t)")
    if "i_minus_meas_t" in results:
        axes[row].plot(t[sl] * 1e6, results["i_minus_meas_t"][sl], label="i- meas(t)", alpha=0.7)
    axes[row].legend()
    axes[row].set_ylabel("Balanced meas.")
    axes[row].grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"Time ($\mu$s)")
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
    t = results["t_ps"]
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
    axes[2].set_xlabel("Time (ps)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_spectra(
    results: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    rbw_mhz: float = 0.0,
    fmin_mhz: Optional[float] = None,
    fmax_mhz: Optional[float] = None,
    loglog: bool = False,
    normalize_to_shot_noise: bool = True,
    shot_noise_source: str = "measured",
    eps: float = 1e-30
) -> plt.Figure:
    f1 = results["freqs_det_mhz"]
    p1 = results["psd_det"]
    f2 = results["freqs_meas_mhz"]
    p2 = results["psd_meas"]

    if rbw_mhz > 0:
        f1, p1 = rbw_average_psd(f1, p1, rbw_mhz)
        f2, p2 = rbw_average_psd(f2, p2, rbw_mhz)

    freqs = f1
    p1 = np.maximum(p1, eps)
    p2 = np.maximum(p2, eps)

    mask = np.ones_like(freqs, dtype=bool)
    if fmin_mhz is not None:
        mask &= freqs >= fmin_mhz
    if fmax_mhz is not None:
        mask &= freqs <= fmax_mhz

    _, _, label = get_primary_detected_channels(results, metadata)

    fig = plt.figure(figsize=(10, 6))

    if normalize_to_shot_noise:
        if shot_noise_source == "measured":
            p_ref = p2
            ref_label = "measured PSD"
        elif shot_noise_source == "deterministic":
            p_ref = p1
            ref_label = "deterministic PSD"
        else:
            raise ValueError(f"Invalid shot noise source: {shot_noise_source}")
        
    trace_det_db = 10.0 * np.log10(p1 / p_ref)
    trace_meas_db = 10.0 * np.log10(p2 / p_ref)

    plt.plot(freqs[mask], trace_det_db[mask], label=f"{label} PSD / {ref_label}")
    plt.plot(freqs[mask], trace_meas_db[mask], label=f"Measured {label} PSD / {ref_label}", alpha=0.7)
    plt.plot(
        freqs[mask],
        np.zeros(np.count_nonzero(mask)),
        "--",
        linewidth=1,
        label=f"Shot noise level ({shot_noise_source})"
    )

    plt.xlabel("Analysis frequency (MHz)")
    plt.ylabel("Noise power (dB)")
    plt.title("Spectrum analyzer trace")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fig.tight_layout()
    return fig


def plot_cumulative_band_power(results: Dict[str, np.ndarray], use_measured: bool = True) -> plt.Figure:
    f = results["freqs_meas_mhz"] if use_measured else results["freqs_det_mhz"]
    p = results["psd_meas"] if use_measured else results["psd_det"]
    df = np.diff(f)
    if len(df) == 0:
        raise ValueError("Not enough frequency points.")
    cum = np.zeros_like(f)
    cum[1:] = np.cumsum(0.5 * (p[1:] + p[:-1]) * df)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(f, cum)
    plt.xlabel("Upper integration frequency (MHz)")
    plt.ylabel("Integrated noise power")
    plt.title("Cumulative integrated PSD")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_all_balanced_psds(
    results: Dict[str, np.ndarray],
    rbw_mhz: float = 10.0,
    fmin_mhz: Optional[float] = None,
    fmax_mhz: Optional[float] = None,
    loglog: bool = False,
    shot_noise_channel: str = "i_minus_det_t_wn",
    shot_noise_channel_ref: str = "i_minus_det_ref_t",
    eps: float = 1e-30
) -> Optional[Dict[str, plt.Figure]]:
    """
    Plot i1, i2, i+, i- measured PSD's if available.
    Useful only for balanced detection

    Output unit:
        10*log10(PSD / shot_noise_level) if shot_noise_channel is available and valid
    
    Shot_noise_channel:
        Channel used as shot-noise reference. Default: "i_minus_meas_t"
    eps:
        Small value to avoid log of zero when normalizing by shot noise level.
    """
    needed = ["i_plus_meas_t", "i_minus_meas_t", "i_plus_det_t_wn", "i_minus_det_t_wn"]
    needed_ref = ["i_plus_det_ref_t", "i_minus_det_ref_t"]
    if not all(k in results for k in needed):
        return None
    
    fs = infer_sample_rate_mhz(results)

    # Compute PSDs for all measured balanced channels
    curves = {}
    for key in needed:
        f, p = compute_psd(results[key], fs_mhz=fs)
        if rbw_mhz > 0:
            f, p = rbw_average_psd(f, p, rbw_mhz)
        curves[key] = (f, p)
    
    for key in needed_ref:
        f, p = compute_psd(results[key], fs_mhz=fs)
        if rbw_mhz > 0:
            f, p = rbw_average_psd(f, p, rbw_mhz)
        curves[key] = (f, p)

    # Shot-noise reference
    if shot_noise_channel not in curves:
        raise ValueError(f"Shot noise reference channel '{shot_noise_channel}' not found among computed PSDs.")
    if shot_noise_channel_ref not in curves:
        raise ValueError(f"Shot noise reference without cavity channel '{shot_noise_channel_ref}' not found among computed PSDs.")
    
    f_ref, p_ref = curves[shot_noise_channel]
    f_ref_ref, p_ref_ref = curves[shot_noise_channel_ref]
    p_ref = np.maximum(p_ref, eps)  # avoid zero or negative values
    p_ref_ref = np.maximum(p_ref_ref, eps)

    # Plot in dB relative to shot noise level
    fig = plt.figure(figsize=(10, 6))

    pretty_labels = {
        "i1_meas_t": "i1",
        "i2_meas_t": "i2",
        "i_plus_meas_t": "i+",
        "i_minus_meas_t": "i-",
        "i_plus_det_ref_t": "i+ det ref",
        "i_minus_det_ref_t": "i- det ref",
        "i_plus_det_t_wn": "i+ det wn",
        "i_minus_det_t_wn": "i- det wn",
    }

    for key in needed:
        f, p = curves[key]

        ratio = np.maximum(p, eps) / p_ref
        trace_db = 10.0 * np.log10(ratio)

        mask = np.ones_like(f, dtype=bool)
        if fmin_mhz is not None:
            mask &= f >= fmin_mhz
        if fmax_mhz is not None:
            mask &= f <= fmax_mhz

        plt.plot(f[mask], trace_db[mask], label=pretty_labels.get(key, key))

    for key in needed_ref:
        f, p = curves[key]

        ratio = np.maximum(p, eps) / p_ref_ref
        trace_db = 10.0 * np.log10(ratio)

        mask = np.ones_like(f, dtype=bool)
        if fmin_mhz is not None:
            mask &= f >= fmin_mhz
        if fmax_mhz is not None:
            mask &= f <= fmax_mhz

        plt.plot(f[mask], trace_db[mask], label=pretty_labels.get(key, key))

        if key == "i_plus_det_ref_t":
            print(np.mean(trace_db[mask]))
            print(trace_db[mask].size)

    # 0 dB shot-noise reference line
    mask_ref = np.ones_like(f_ref, dtype=bool)
    if fmin_mhz is not None:
        mask_ref &= f_ref >= fmin_mhz
    if fmax_mhz is not None:
        mask_ref &= f_ref <= fmax_mhz
    
    #plt.plot(
        #f_ref[mask_ref],
        #np.zeros(np.count_nonzero(mask_ref)),
        #"--",
        #linewidth=1,
        #label=f"Shot noise level ({shot_noise_channel})"
    #)
    
    plt.xlabel("Analysis frequency (MHz)")
    plt.ylabel("Noise power (dB)")
    plt.title("PSD of balanced detection channels")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    fig.tight_layout()
    return fig

    
def plot_lo_phase_sweep_from_saved_signal(
    results: Dict[str, np.ndarray],
    n_phases: int = 41,
    use_psd_at_mhz: Optional[float] = None,
    band: Optional[Tuple[Optional[float], Optional[float]]] = None,
    nperseg: int = 2**14,
) -> plt.Figure:
    if "s_out_t" not in results:
        raise ValueError("Need s_out_t to reconstruct a LO-phase sweep.")

    z = results["s_out_t"]
    fs = infer_sample_rate_mhz(results)
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases)
    values = []

    for theta in phases:
        i_theta = quadrature_projection(z, theta)
        if use_psd_at_mhz is None and band is None:
            values.append(float(np.var(i_theta)))
        else:
            f, p = compute_psd(i_theta, fs_mhz=fs, nperseg=nperseg)
            if use_psd_at_mhz is not None:
                values.append(psd_value_at(f, p, use_psd_at_mhz))
            else:
                fmin_mhz, fmax_mhz = band if band is not None else (None, None)
                values.append(integrated_band_power(f, p, fmin_mhz, fmax_mhz))

    fig = plt.figure(figsize=(8, 5))
    plt.plot(phases, values, marker="o")
    plt.xlabel("LO phase θ (rad)")
    if use_psd_at_mhz is None and band is None:
        plt.ylabel("Var[homodyne current]")
        plt.title("LO phase sweep from saved output field")
    elif use_psd_at_mhz is not None:
        plt.ylabel(f"PSD at {use_psd_at_mhz:.3g} MHz")
        plt.title("LO phase sweep at fixed analysis frequency")
    else:
        fmin_mhz, fmax_mhz = band if band is not None else (None, None)
        plt.ylabel("Integrated PSD")
        plt.title(f"LO phase sweep integrated over {fmin_mhz:.3g}–{fmax_mhz:.3g} MHz")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_bistability(results: Dict[str, np.ndarray]) -> Optional[plt.Figure]:
    needed = [
        "bistab_F_up",
        "bistab_density_up",
        "bistab_F_down",
        "bistab_density_down",
    ]

    if not all(k in results for k in needed):
        print("No bistability data found in saved results.")
        return None

    fig = plt.figure(figsize=(7, 5))

    plt.scatter(np.real(results["bistab_F_up"]), results["bistab_density_up"],
                label="Sweep up", marker="x")
    plt.scatter(np.real(results["bistab_F_down"]), results["bistab_density_down"],
                label="Sweep down", marker="+")

    if "F_work" in results:
        F_work = np.real(results["F_work"][0])

        if "rho_work" in results:
            rho_work = float(results["rho_work"][0])
        elif "psi_t" in results:
            rho_work = float(np.mean(np.abs(results["psi_t"])**2))
        else:
            rho_work = np.nan

    plt.scatter(
        [F_work],
        [rho_work],
        s=60,
        color="red",
        label="Working point",
        zorder=5,
    )

    plt.xlabel("Pump amplitude F")
    plt.ylabel(r"Intracavity density $|\psi|^2$")
    plt.title("Polariton bistability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    return fig

def plot_kerneldensityestimation(
    results,
    gridsize=300,
    cmap="magma",
    remove_mean=True,
):
    from scipy.stats import gaussian_kde

    datasets = [
        ("Input", results["x_in"], results["p_in"], "X_in", "P_in"),
        ("Intracavity", results["x_cav"], results["p_cav"], "X_cav", "P_cav"),
        ("Output", results["x_out"], results["p_out"], "X_out", "P_out"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    last_im = None

    for ax, (title, x, p, xlabel, ylabel) in zip(axes, datasets):
        x = np.asarray(x, dtype=float)
        p = np.asarray(p, dtype=float)

        if remove_mean:
            x = x - np.mean(x)
            p = p - np.mean(p)

        values = np.vstack([x, p])

        try:
            kde = gaussian_kde(values)
        except np.linalg.LinAlgError:
            values = values + 1e-8 * np.random.normal(size=values.shape)
            kde = gaussian_kde(values)

        x_min, x_max = np.percentile(x, [0.5, 99.5])
        p_min, p_max = np.percentile(p, [0.5, 99.5])

        # évite extent nul si P_in est constant
        if abs(x_max - x_min) < 1e-12:
            x_min -= 1e-6
            x_max += 1e-6
        if abs(p_max - p_min) < 1e-12:
            p_min -= 1e-6
            p_max += 1e-6

        X, P = np.meshgrid(
            np.linspace(x_min, x_max, gridsize),
            np.linspace(p_min, p_max, gridsize),
        )

        positions = np.vstack([X.ravel(), P.ravel()])
        density = kde(positions).reshape(X.shape)
        density /= np.max(density)

        last_im = ax.imshow(
            density,
            origin="lower",
            extent=[x_min, x_max, p_min, p_max],
            aspect="auto",
            cmap=cmap,
            interpolation="bilinear",
            vmin=0,
            vmax=1,
        )

        ax.set_title(title, fontsize=20)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Normalized density", fontsize=14)

    plt.show()

def plot_output_noise_vs_input_noise(results: Dict[str, np.ndarray]) -> Optional[plt.Figure]:

    needed = [
        "transfer_var_xin",
        "transfer_var_xout",
        "transfer_var_pout",
    ]

    if not all(k in results for k in needed):
        print("No transfer scan data found.")
        return None

    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        results["transfer_var_xin"],
        results["transfer_var_xout"],
        "o",
        ms=4,
        label=r"$X_{in}\rightarrow X_{out}$"
    )

    plt.plot(
        results["transfer_var_xin"],
        results["transfer_var_pout"],
        "o",
        ms=4,
        label=r"$X_{in}\rightarrow P_{out}$"
    )

    plt.xlabel(r"Var($X_{in}$)")
    plt.ylabel("Output variance")
    plt.title("Output noise versus input amplitude noise")
    plt.grid(True, alpha=0.3)
    plt.legend()

    fig.tight_layout()
    return fig


def plot_transfer_gain(results: Dict[str, np.ndarray]) -> Optional[plt.Figure]:

    needed = [
        "transfer_gains_dB",
        "transfer_G_Xin_to_Xout",
        "transfer_G_Xin_to_Pout",
    ]

    if not all(k in results for k in needed):
        print("No transfer gain data found.")
        return None

    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        results["transfer_gains_dB"],
        results["transfer_G_Xin_to_Xout"],
        "o",
        ms=4,
        label=r"$G_{X\to X}$"
    )

    plt.plot(
        results["transfer_gains_dB"],
        results["transfer_G_Xin_to_Pout"],
        "o",
        ms=4,
        label=r"$G_{X\to P}$"
    )

    plt.axvline(
        5,
        linestyle="--",
        alpha=0.7,
        label="experiment: 5 dB"
    )

    plt.xlabel("Input amplitude noise gain (dB)")
    plt.ylabel("Variance transfer gain")
    plt.title("Transfer gain versus input noise gain")
    plt.grid(True, alpha=0.3)
    plt.legend()

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
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Results/polariton_homodyne_results_sweep_phase_noise.npz")
    save_figures = True
    output_dir = Path("Plots/polariton_homodyne_results_sweep_phase_noise")

    # Choose what to replot and how
    time_trace = True
    quadratures_vs_time = True
    phase_space = False
    spectra = False
    cumulative = False
    psds = False
    lo_sweep = False
    bistability = True
    kde_plot = True
    transfer_noise = True
    transfer_gain = True

    # spectrum display choices
    fmin_mhz = 1e-3
    fmax_mhz = 2000.0
    rbw_mhz = 4.0
    loglog = False

    # LO phase sweep replot choices
    lo_sweep_phases = 41
    lo_sweep_use_psd_at_mhz = 1      # set to None to disable fixed-frequency PSD mode
    lo_sweep_band = None              # example: (5e5, 2e6). If not None, used instead of var().

    # -----------------------
    # Load
    # -----------------------
    metadata, results = load_results_npz(input_path)

    # Add the path to the metadata for provenance
    metadata.setdefault("replot_info", {})
    metadata["replot_info"]["source_file"] = str(input_path.resolve())
    results = build_missing_channels_from_output(results, metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata_replot.json"

    with open(meta_path, "w", encoding="utf-8") as f:
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

    if time_trace:
        figs["time_traces.png"] = plot_time_traces(results, metadata)
    if quadratures_vs_time:
        figs["quadratures_vs_time.png"] = plot_quadratures_vs_time(results)
    if phase_space:
        figs["phase_space.png"] = plot_phase_space(results)
    if spectra:
        figs["spectra.png"] = plot_spectra(results, metadata, rbw_mhz=rbw_mhz, fmin_mhz=fmin_mhz, fmax_mhz=fmax_mhz, loglog=loglog)
    if cumulative:
        figs["cumulative_band_power.png"] = plot_cumulative_band_power(results, use_measured=True)
    if bistability:
        fig_bistab = plot_bistability(results)
        if fig_bistab is not None:
            figs["bistability.png"] = fig_bistab
    if kde_plot:
        figs["quadrature_kde.png"] = plot_kerneldensityestimation(
            results,
            remove_mean=True
        )
    if transfer_noise:
        fig_transfer_noise = plot_output_noise_vs_input_noise(results)
        if fig_transfer_noise is not None:
            figs["output_noise_vs_input_noise.png"] = fig_transfer_noise

    if transfer_gain:
        fig_transfer_gain = plot_transfer_gain(results)
        if fig_transfer_gain is not None:
            figs["transfer_gain.png"] = fig_transfer_gain


    balanced_psd_fig = plot_all_balanced_psds(results, rbw_mhz=rbw_mhz, fmin_mhz=fmin_mhz, fmax_mhz=fmax_mhz, loglog=loglog)

    if balanced_psd_fig is not None and psds:
        figs["balanced_psds.png"] = balanced_psd_fig

    if "s_out_t" in results and lo_sweep:
        figs["lo_phase_sweep_var.png"] = plot_lo_phase_sweep_from_saved_signal(
            results,
            n_phases=lo_sweep_phases,
            use_psd_at_mhz=None,
            band=None,
        )

        if lo_sweep_use_psd_at_mhz is not None:
            figs["lo_phase_sweep_psd_fixed_freq.png"] = plot_lo_phase_sweep_from_saved_signal(
                results,
                n_phases=lo_sweep_phases,
                use_psd_at_mhz=lo_sweep_use_psd_at_mhz,
                band=None,
            )

        if lo_sweep_band is not None:
            figs["lo_phase_sweep_psd_band.png"] = plot_lo_phase_sweep_from_saved_signal(
                results,
                n_phases=lo_sweep_phases,
                use_psd_at_mhz=None,
                band=lo_sweep_band,
            )

    if save_figures:
        for name, fig in figs.items():
            save_figure(fig, output_dir, name)
        print(f"\nSaved figures to: {output_dir.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()