#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amplitude-only reconstruction of input noise from a polariton microcavity output.

This script is designed to live next to `Polariton_Microcavity_OHT.py` in the
Thermal-states repository. It reuses the existing classes and functions from that
file, and only adds the dataset generation + regression layer.

Physical goal
-------------
We drive the polariton cavity with a noisy complex envelope

    F(t) = F_s + F_n(t)

where F_n can be amplitude noise, phase noise, or both. The cavity can amplify
and rotate/transfer noise between quadratures. Experimentally, we assume a
balanced direct detection without LO, so we only use an amplitude/intensity-like
measurement channel at the output. We then ask whether this single measured
channel is sufficient to reconstruct the input-noise covariance:

    Var(X_in), Var(P_in), Cov(X_in, P_in)

for three cases:
    amplitude noise only, phase noise only, both.

Usage examples
--------------
From the repo folder:

    python amplitude_only_reconstruction_from_OHT.py --case amplitude --plot
    python amplitude_only_reconstruction_from_OHT.py --case phase --plot
    python amplitude_only_reconstruction_from_OHT.py --case both --plot
    python amplitude_only_reconstruction_from_OHT.py --case all --plot

Notes
-----
- The measured channel is chosen by --measurement-channel.
- For balanced direct detection without LO, the most natural channel in your code
  is usually `i_plus_meas_t`, because it contains intensity noise. Around a strong
  coherent mean field, intensity fluctuations are approximately proportional to
  the amplitude quadrature of the output field.
- You can also choose `x_out` to perform an idealized amplitude-quadrature test,
  useful as a best-case benchmark.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, Literal, Optional, Tuple
import argparse
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# -----------------------------------------------------------------------------
# Imports from your existing simulation code.
# Keep this file in the same folder as Polariton_Microcavity_OHT.py.
# -----------------------------------------------------------------------------
try:
    from Polariton_Microcavity_OHT import (
        FullConfig,
        NoiseConfig,
        CavityConfig,
        SimulationConfig,
        DetectionConfig,
        SpectrumConfig,
        clone_config,
        set_input_noise_gain,
        time_axis,
        complex_to_quadratures,
        generate_drive_noise,
        integrate_cavity,
        output_field,
        balanced_direct_detection_currents,
        balanced_direct_detection_currents_without_noise,
        compute_psd,
        choose_pump_values_from_bistability,
        prepare_upper_branch,
        MHZ_PER_INV_PS,
    )
except ImportError as exc:
    raise ImportError(
        "Could not import from Polariton_Microcavity_OHT.py. "
        "Put this script in the same folder as Polariton_Microcavity_OHT.py "
        "or add the repo folder to PYTHONPATH."
    ) from exc


NoiseCase = Literal["amplitude", "phase", "both"]
MeasurementChannel = Literal["i_plus", "i_minus", "x_out"]


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def central_moment_features(x: np.ndarray) -> np.ndarray:
    """Features from the measured amplitude-only time trace."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        raise ValueError("Not enough points in measured trace to build features.")

    mean = float(np.mean(x))
    xc = x - mean
    var = float(np.mean(xc**2))
    std = math.sqrt(max(var, 1e-30))

    skew = float(np.mean(xc**3) / (std**3 + 1e-30))
    kurt = float(np.mean(xc**4) / (std**4 + 1e-30))
    return np.array([mean, var], dtype=float)
    # return np.array([mean, var, std, skew, kurt], dtype=float)


def psd_band_features(
    x: np.ndarray,
    fs_mhz: float,
    spectrum_cfg: SpectrumConfig,
    n_bands: int = 12,
    fmin_mhz: float = 0.0,
    fmax_mhz: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSD-band features from a measured amplitude-only trace.

    Returns
    -------
    features : array
        Integrated powers in frequency bands + global spectral summaries.
    freqs : array
        Frequencies of the PSD.
    psd : array
        PSD values.
    """
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    freqs, psd = compute_psd(x, fs_mhz, spectrum_cfg)

    if fmax_mhz is None:
        fmax_mhz = float(np.max(freqs))
    mask = (freqs >= fmin_mhz) & (freqs <= fmax_mhz)
    if not np.any(mask):
        raise ValueError("Empty PSD frequency window. Check fmin/fmax.")

    f = freqs[mask]
    p = np.maximum(psd[mask], 1e-300)

    edges = np.linspace(float(f[0]), float(f[-1]), n_bands + 1)
    band_powers = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        if not np.any(m):
            band_powers.append(0.0)
        else:
            band_powers.append(float(np.trapz(p[m], f[m])))

    total = float(np.trapz(p, f))
    centroid = float(np.trapz(f * p, f) / (total + 1e-300))
    log_mean = float(np.mean(np.log10(p)))
    log_std = float(np.std(np.log10(p)))

    #features = np.array([*band_powers, total, centroid, log_mean, log_std], dtype=float)
    features = np.array([total], dtype=float)
    return features, freqs, psd


def amplitude_only_features(
    measured: np.ndarray,
    fs_mhz: float,
    spectrum_cfg: SpectrumConfig,
    n_bands: int = 12,
    fmin_mhz: float = 0.0,
    fmax_mhz: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Combine time-domain and spectrum-domain features."""
    mom = central_moment_features(measured)
    spec, freqs, psd = psd_band_features(
        measured,
        fs_mhz=fs_mhz,
        spectrum_cfg=spectrum_cfg,
        n_bands=n_bands,
        fmin_mhz=fmin_mhz,
        fmax_mhz=fmax_mhz,
    )
    return np.concatenate([mom, spec]), {"freqs_mhz": freqs, "psd": psd}


def input_noise_targets(F_t: np.ndarray) -> np.ndarray:
    """
    Regression targets: covariance of the input drive quadratures.

    We subtract the coherent mean because the goal is to reconstruct the noise,
    not the DC coherent field.
    """
    x_in, p_in = complex_to_quadratures(F_t)
    x = x_in - np.mean(x_in)
    p = p_in - np.mean(p_in)
    cov = np.cov(x, p, ddof=0)
    return np.array([cov[0, 0], cov[1, 1], cov[0, 1]], dtype=float)


def select_measured_channel(
    results: Dict[str, np.ndarray],
    channel: MeasurementChannel,
) -> np.ndarray:
    """Choose the experimental amplitude-only measurement channel."""
    if channel == "i_plus":
        return np.asarray(results["i_plus_meas_t"], dtype=float)
    if channel == "i_minus":
        return np.asarray(results["i_minus_meas_t"], dtype=float)
    if channel == "x_out":
        # Idealized amplitude quadrature, for comparison with the real detector.
        return np.asarray(results["x_out"], dtype=float)
    raise ValueError(f"Unknown measurement channel: {channel}")


# -----------------------------------------------------------------------------
# Simulation layer reusing Polariton_Microcavity_OHT.py
# -----------------------------------------------------------------------------

def simulate_one_sample(
    base_cfg: FullConfig,
    case: NoiseCase,
    seed: int,
    gain_amp_db: float,
    gain_phase_db: float,
    F_work: complex,
    psi_initial: complex,
    measurement_channel: MeasurementChannel,
    n_bands: int,
    fmin_mhz: float,
    fmax_mhz: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Run one noisy cavity simulation and return features/targets/results."""
    cfg = clone_config(base_cfg)

    cfg.noise.mode = case
    cfg.noise.seed = int(seed)
    cfg.cavity.F_s = F_work
    cfg.cavity.psi0 = psi_initial
    cfg.detection.mode = "balanced_sum"

    # Set noise strengths according to the chosen case.
    if case == "amplitude":
        set_input_noise_gain(cfg, gain_dB_amp=gain_amp_db, gain_dB_phase=0.0)
        cfg.noise.strength_phase = 0.0
    elif case == "phase":
        set_input_noise_gain(cfg, gain_dB_amp=0.0, gain_dB_phase=gain_phase_db)
        cfg.noise.strength_amp = 0.0
    elif case == "both":
        set_input_noise_gain(cfg, gain_dB_amp=gain_amp_db, gain_dB_phase=gain_phase_db)
    else:
        raise ValueError(f"Unknown case: {case}")

    t_full = time_axis(cfg.sim.duration_ps, cfg.sim.dt_ps)
    dt_ps = cfg.sim.dt_ps
    fs_mhz = (1.0 / dt_ps) * MHZ_PER_INV_PS

    # Use existing noise generation and cavity evolution from OHT.
    F_n, noise_aux = generate_drive_noise(t_full, cfg.noise)
    F_t = cfg.cavity.F_s + F_n
    psi_t = integrate_cavity(t_full, F_t, cfg.cavity, integrator=cfg.sim.integrator)
    s_out_t = output_field(F_t, psi_t, cfg.cavity)

    # Existing balanced direct detection from OHT.
    rng = np.random.default_rng(seed + 999)
    det = balanced_direct_detection_currents(
        s_out_t=s_out_t,
        cfg=cfg.detection,
        rng=rng,
        dt_ps=dt_ps,
    )

    x_in, p_in = complex_to_quadratures(F_t)
    x_out, p_out = complex_to_quadratures(s_out_t)

    results: Dict[str, np.ndarray] = {
        "t_ps": t_full,
        "F_t": F_t,
        "psi_t": psi_t,
        "s_out_t": s_out_t,
        "amp_noise": noise_aux["amp_noise"],
        "phase_noise": noise_aux["phase_noise"],
        "x_in": x_in,
        "p_in": p_in,
        "x_out": x_out,
        "p_out": p_out,
        **det,
    }

    # Discard transient and downsample exactly in the spirit of OHT.
    n0 = int(cfg.sim.discard_fraction * t_full.size)
    step = max(1, cfg.sim.store_every)
    for k in list(results.keys()):
        arr = results[k]
        if isinstance(arr, np.ndarray) and arr.shape == t_full.shape:
            results[k] = arr[n0::step]

    fs_store_mhz = fs_mhz / step
    measured = select_measured_channel(results, measurement_channel)
    features, aux = amplitude_only_features(
        measured,
        fs_mhz=fs_store_mhz,
        spectrum_cfg=cfg.spectrum,
        n_bands=n_bands,
        fmin_mhz=fmin_mhz,
        fmax_mhz=fmax_mhz,
    )
    target = input_noise_targets(results["F_t"])

    results["measured_t"] = measured
    results["feature_freqs_mhz"] = aux["freqs_mhz"]
    results["feature_psd"] = aux["psd"]
    results["target_var_xin"] = np.array([target[0]])
    results["target_var_pin"] = np.array([target[1]])
    results["target_cov_xpin"] = np.array([target[2]])

    return features, target, results


# -----------------------------------------------------------------------------
# Dataset + regression
# -----------------------------------------------------------------------------

def make_gain_grid(
    rng: np.random.Generator,
    case: NoiseCase,
    n_samples: int,
    gain_min_db: float,
    gain_max_db: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw input noise gains for amplitude and phase quadratures."""
    if case == "amplitude":
        ga = rng.uniform(gain_min_db, gain_max_db, size=n_samples)
        gp = np.zeros(n_samples)
    elif case == "phase":
        ga = np.zeros(n_samples)
        gp = rng.uniform(gain_min_db, gain_max_db, size=n_samples)
    elif case == "both":
        ga = rng.uniform(gain_min_db, gain_max_db, size=n_samples)
        gp = rng.uniform(gain_min_db, gain_max_db, size=n_samples)
    else:
        raise ValueError(case)
    return ga, gp


def build_dataset(
    base_cfg: FullConfig,
    case: NoiseCase,
    n_samples: int,
    seed0: int,
    gain_min_db: float,
    gain_max_db: float,
    F_work: complex,
    psi_initial: complex,
    measurement_channel: MeasurementChannel,
    n_bands: int,
    fmin_mhz: float,
    fmax_mhz: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed0)
    gain_amp, gain_phase = make_gain_grid(rng, case, n_samples, gain_min_db, gain_max_db)
    seeds = rng.integers(1, 2**31 - 1, size=n_samples)

    X_list = []
    Y_list = []
    for i in range(n_samples):
        feat, target, _ = simulate_one_sample(
            base_cfg=base_cfg,
            case=case,
            seed=int(seeds[i]),
            gain_amp_db=float(gain_amp[i]),
            gain_phase_db=float(gain_phase[i]),
            F_work=F_work,
            psi_initial=psi_initial,
            measurement_channel=measurement_channel,
            n_bands=n_bands,
            fmin_mhz=fmin_mhz,
            fmax_mhz=fmax_mhz,
        )
        X_list.append(feat)
        Y_list.append(target)
        print(
            f"[{case}] sample {i+1:03d}/{n_samples:03d} "
            f"gain_amp={gain_amp[i]:5.2f} dB gain_phase={gain_phase[i]:5.2f} dB "
            f"target=[{target[0]:.3e}, {target[1]:.3e}, {target[2]:.3e}]"
        )

    meta = {
        "gain_amp_db": gain_amp,
        "gain_phase_db": gain_phase,
        "seeds": seeds,
    }
    return np.vstack(X_list), np.vstack(Y_list), meta


def fit_ridge(X_train: np.ndarray, Y_train: np.ndarray, alpha: float):
    if SKLEARN_AVAILABLE:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(X_train, Y_train)
        return model

    # Minimal fallback if sklearn is not installed.
    class SimpleRidge:
        def fit(self, X, Y):
            self.x_mean = X.mean(axis=0)
            self.x_std = X.std(axis=0) + 1e-12
            self.y_mean = Y.mean(axis=0)
            Z = (X - self.x_mean) / self.x_std
            A = Z.T @ Z + alpha * np.eye(Z.shape[1])
            B = Z.T @ (Y - self.y_mean)
            self.W = np.linalg.solve(A, B)
            return self

        def predict(self, X):
            Z = (X - self.x_mean) / self.x_std
            return self.y_mean + Z @ self.W

    return SimpleRidge().fit(X_train, Y_train)


def evaluate_predictions(Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    names = ["Var_X_in", "Var_P_in", "Cov_XP_in"]
    r2 = []
    rmse = []
    for j in range(Y_true.shape[1]):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2_j = 1.0 - ss_res / (ss_tot + 1e-30)
        r2.append(r2_j)
        rmse.append(math.sqrt(float(np.mean((yt - yp) ** 2))))
    return {"names": np.array(names), "r2": np.array(r2), "rmse": np.array(rmse)}


def plot_predictions(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    case: NoiseCase,
    outdir: Path,
) -> None:
    names = [r"Var$(X_{in})$", r"Var$(P_{in})$", r"Cov$(X_{in},P_{in})$"]
    for j, name in enumerate(names):
        fig = plt.figure(figsize=(5.5, 5))
        plt.scatter(Y_true[:, j], Y_pred[:, j], s=28, alpha=0.75)
        lo = min(float(np.min(Y_true[:, j])), float(np.min(Y_pred[:, j])))
        hi = max(float(np.max(Y_true[:, j])), float(np.max(Y_pred[:, j])))
        if abs(hi - lo) < 1e-20:
            lo -= 1e-12
            hi += 1e-12
        plt.plot([lo, hi], [lo, hi], "--", linewidth=1)
        plt.xlabel("True " + name)
        plt.ylabel("Predicted " + name)
        plt.title(f"Amplitude-only reconstruction: {case}")
        plt.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"prediction_{case}_{j}.png", dpi=180)

# -----------------------------------------------------------------------------
# High-level run
# -----------------------------------------------------------------------------

def prepare_working_point(
    cfg: FullConfig,
    use_upper_branch: bool,
    F_work_manual: Optional[float],
) -> Tuple[complex, complex, Dict[str, float]]:
    """Return (F_work, psi_initial, metadata)."""
    if F_work_manual is not None:
        F_work = complex(F_work_manual, 0.0)
        # Quick deterministic relaxation to get a reasonable steady initial state.
        t = time_axis(min(cfg.sim.duration_ps, 2e4), cfg.sim.dt_ps)
        old = cfg.cavity.psi0
        cfg.cavity.psi0 = 0.0 + 0.0j
        psi = integrate_cavity(
            t,
            np.full(t.shape, F_work, dtype=np.complex128),
            cfg.cavity,
            integrator=cfg.sim.integrator,
        )
        cfg.cavity.psi0 = old
        return F_work, psi[-1], {"F_work": float(F_work.real), "upper_branch": False}

    if use_upper_branch:
        pump = choose_pump_values_from_bistability(
            cfg,
            F_min=0.0,
            F_max=2.0,
            n_F=80,
            alpha_work=0.9,
        )
        F_work = pump["F_work"]
        psi_initial = prepare_upper_branch(
            cfg,
            F_low=pump["F_low"],
            F_high=pump["F_high"],
            F_work=F_work,
        )
        return F_work, psi_initial, {
            "F_low": float(np.real(pump["F_low"])),
            "F_high": float(np.real(pump["F_high"])),
            "F_work": float(np.real(F_work)),
            "F_left": float(pump["F_left"]),
            "F_right": float(pump["F_right"]),
            "upper_branch": True,
        }

    F_work = cfg.cavity.F_s
    t = time_axis(min(cfg.sim.duration_ps, 2e4), cfg.sim.dt_ps)
    old = cfg.cavity.psi0
    cfg.cavity.psi0 = 0.0 + 0.0j
    psi = integrate_cavity(
        t,
        np.full(t.shape, F_work, dtype=np.complex128),
        cfg.cavity,
        integrator=cfg.sim.integrator,
    )
    cfg.cavity.psi0 = old
    return F_work, psi[-1], {"F_work": float(np.real(F_work)), "upper_branch": False}


def run_case(
    case: NoiseCase,
    args: argparse.Namespace,
    base_cfg: FullConfig,
    F_work: complex,
    psi_initial: complex,
    outdir: Path,
) -> Dict[str, np.ndarray]:
    print("\n" + "=" * 80)
    print(f"Case: {case}")
    print("=" * 80)

    features_names = np.array(["mean", "var", "total_power"])

    #features_names = np.array(
        #["mean", "var", "std", "skew", "kurt"] 
        #+ [f"band_{i}" for i in range(args.n_bands)] 
        #+ ["total_power", "centroid", "log_mean", "log_std"]
    #)

    X_train, Y_train, meta_train = build_dataset(
        base_cfg=base_cfg,
        case=case,
        n_samples=args.n_train,
        seed0=args.seed,
        gain_min_db=args.gain_min_db,
        gain_max_db=args.gain_max_db,
        F_work=F_work,
        psi_initial=psi_initial,
        measurement_channel=args.measurement_channel,
        n_bands=args.n_bands,
        fmin_mhz=args.fmin_mhz,
        fmax_mhz=args.fmax_mhz,
    )
    X_test, Y_test, meta_test = build_dataset(
        base_cfg=base_cfg,
        case=case,
        n_samples=args.n_test,
        seed0=args.seed + 100_000,
        gain_min_db=args.gain_min_db,
        gain_max_db=args.gain_max_db,
        F_work=F_work,
        psi_initial=psi_initial,
        measurement_channel=args.measurement_channel,
        n_bands=args.n_bands,
        fmin_mhz=args.fmin_mhz,
        fmax_mhz=args.fmax_mhz,
    )

    model = fit_ridge(X_train, Y_train, alpha=args.ridge_alpha)
    ridge_model = model.named_steps["ridge"] if SKLEARN_AVAILABLE else model
    Y_pred = model.predict(X_test)
    metrics = evaluate_predictions(Y_test, Y_pred)

    target_names = np.array(metrics["names"])

    ridge_coef = ridge_model.coef_ if SKLEARN_AVAILABLE else model.W
    ridge_intercept = ridge_model.intercept_ if SKLEARN_AVAILABLE else model.y_mean

    ridge_coef_labeled = {
        str(target): {
            str(feature): float(coef) for feature, coef in zip(features_names, coef_row)
        }
        for target, coef_row in zip(target_names, ridge_coef)
    }

    ridge_importance_label = {}
    for target, coef_row in zip(target_names, ridge_coef):
        abs_coef = np.abs(coef_row)
        importance = abs_coef / (abs_coef.sum() + 1e-30)
        ridge_importance_label[str(target)] = {
            str(feature): float(imp) for feature, imp in zip(features_names, importance)
        }

    print("\nReconstruction metrics from amplitude-only detection")
    for name, r2, rmse in zip(metrics["names"], metrics["r2"], metrics["rmse"]):
        print(f"  {name:>10s}: R2 = {r2: .4f}, RMSE = {rmse:.4e}")

    case_out = outdir / case
    case_out.mkdir(parents=True, exist_ok=True)

    if args.plot:
        plot_predictions(Y_test, Y_pred, case, case_out)
        #plt.show()

        for target_name, coef_row in zip(target_names, ridge_coef):
            abs_coef = np.abs(coef_row)
            importance = abs_coef / (abs_coef.sum() + 1e-30)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(features_names, importance)
            ax.set_title(f"Feature importance for predicting {target_name} ({case})")
            ax.set_ylabel("Normalized ridge coefficient magnitude")
            ax.set_xlabel("Feature")
            plt.xticks(rotation=90)
            fig.tight_layout()
            fig.savefig(case_out / f"feature_importance_{case}_{target_name}.png", dpi=180)
        
        #plt.show(block=True)

    np.savez_compressed(
        case_out / f"amplitude_only_reconstruction_{case}.npz",
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        Y_pred=Y_pred,
        train_gain_amp_db=meta_train["gain_amp_db"],
        train_gain_phase_db=meta_train["gain_phase_db"],
        test_gain_amp_db=meta_test["gain_amp_db"],
        test_gain_phase_db=meta_test["gain_phase_db"],
        r2=metrics["r2"],
        rmse=metrics["rmse"],
        ridge_coef_labeled = ridge_coef_labeled,
        ridge_importance_label = ridge_importance_label,
    )
    coef = np.abs(ridge_model.coef_[0]) if SKLEARN_AVAILABLE else np.abs(model.W[0])
    importance = coef / (coef.sum() + 1e-30)
    plt.bar(features_names, importance)
    plt.xticks(rotation=90)

    return {"Y_test": Y_test, "Y_pred": Y_pred, "r2": metrics["r2"], "rmse": metrics["rmse"],
            "n_train": args.n_train, "n_test": args.n_test, "ridge_alpha": args.ridge_alpha, "measurement_channel": args.measurement_channel,
            "feature_names": features_names, "target_names": target_names, "ridge_coef_labeled": ridge_coef_labeled, "ridge_importance_label": ridge_importance_label, 
            "ridge_intercept": ridge_intercept
            }


def make_base_config(args: argparse.Namespace) -> FullConfig:
    cfg = FullConfig()

    # Keep defaults from Polariton_Microcavity_OHT.py, but allow quick CLI changes.
    cfg.sim.duration_ps = args.duration_ps
    cfg.sim.dt_ps = args.dt_ps
    cfg.sim.store_every = args.store_every
    cfg.sim.discard_fraction = args.discard_fraction
    cfg.sim.integrator = args.integrator

    cfg.noise.cutoff_mhz = args.cutoff_mhz
    cfg.noise.seed = args.seed

    cfg.detection.mode = "balanced_sum"
    cfg.detection.simulate_vacuum_port = args.simulate_vacuum_port
    cfg.detection.sigma_vac = args.sigma_vac
    cfg.detection.detection_efficiency = args.detection_efficiency
    cfg.detection.responsivity = args.responsivity

    cfg.spectrum.nperseg = args.nperseg
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Amplitude-only reconstruction using Polariton_Microcavity_OHT.py functions/classes."
    )
    p.add_argument("--case", choices=["amplitude", "phase", "both", "all"], default="all")
    p.add_argument("--measurement-channel", choices=["i_plus", "i_minus", "x_out"], default="i_plus")

    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--ridge-alpha", type=float, default=1e-2)

    p.add_argument("--gain-min-db", type=float, default=0.0)
    p.add_argument("--gain-max-db", type=float, default=25.0)
    p.add_argument("--cutoff-mhz", type=float, default=2500.0)

    p.add_argument("--duration-ps", type=float, default=4e2)
    p.add_argument("--dt-ps", type=float, default=1.0)
    p.add_argument("--store-every", type=int, default=10)
    p.add_argument("--discard-fraction", type=float, default=0.2)
    p.add_argument("--integrator", choices=["rk4", "heun", "euler"], default="rk4")
    p.add_argument("--nperseg", type=int, default=2**10)

    p.add_argument("--n-bands", type=int, default=12)
    p.add_argument("--fmin-mhz", type=float, default=0.0)
    p.add_argument("--fmax-mhz", type=float, default=None)

    p.add_argument("--sigma-vac", type=float, default=0.02)
    p.add_argument("--detection-efficiency", type=float, default=1.0)
    p.add_argument("--responsivity", type=float, default=1.0)
    p.add_argument("--simulate-vacuum-port", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--upper-branch", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--F-work", type=float, default=None, help="Manual working pump amplitude. Overrides --upper-branch.")

    p.add_argument("--outdir", type=str, default="Results/amplitude_only_reconstruction6")
    p.add_argument("--plot", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = make_base_config(args)
    F_work, psi_initial, wp_meta = prepare_working_point(
        cfg,
        use_upper_branch=args.upper_branch,
        F_work_manual=args.F_work,
    )

    print("Working point:")
    print(json.dumps(wp_meta, indent=2))
    print(f"Initial density |psi0|^2 = {abs(psi_initial)**2:.6g}")
    print(f"Measurement channel = {args.measurement_channel}")

    cases: Iterable[NoiseCase]
    if args.case == "all":
        cases = ["amplitude", "phase", "both"]
    else:
        cases = [args.case]

    summary = {}
    for case in cases:
        summary[case] = run_case(case, args, cfg, F_work, psi_initial, outdir)

    # Save a compact JSON summary.
    json_summary = {}
    for case, data in summary.items():
        json_summary[case] = {
            "r2_Var_X_in": float(data["r2"][0]),
            "r2_Var_P_in": float(data["r2"][1]),
            "r2_Cov_XP_in": float(data["r2"][2]),
            "rmse_Var_X_in": float(data["rmse"][0]),
            "rmse_Var_P_in": float(data["rmse"][1]),
            "rmse_Cov_XP_in": float(data["rmse"][2]),
            "n_train": int(data["n_train"]),
            "n_test": int(data["n_test"]),
            "ridge_alpha": float(data["ridge_alpha"]),
            "measurement_channel": str(data["measurement_channel"]),
            "feature_names": data["feature_names"].tolist(),
            "target_names": data["target_names"].tolist(),
            "ridge_coef_labeled": data["ridge_coef_labeled"],
            "ridge_importance_label": data["ridge_importance_label"],
            "ridge_intercept": data["ridge_intercept"].tolist(), 
        }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"working_point": wp_meta, "summary": json_summary}, f, indent=2)

    print("\nSaved results in:", outdir)
    print("Summary:")
    print(json.dumps(json_summary, indent=2))


if __name__ == "__main__":
    main()
