"""
train_noise_to_output_bistability.py

Small learning pipeline built on top of Polariton_Microcavity_bistability.py.

Goal
----
Generate many noisy polariton-microcavity simulations where only the amplitude
quadrature noise strength sigma_amp is varied, extract output-noise features,
train a simple regression model, and test whether the output noise lets us
recover the injected input sigma_amp.


Run for a quick test:
    python train_noise_to_output_bistability.py --quick --show

Run a more serious dataset:
    python train_noise_to_output_bistability.py --n-train 30 --n-test 15 --repeats 3 --show

The script saves:
    learning_bistability_results.npz
    Plots_learning_bistability/*.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from Polariton_Microcavity_bistability import (
    FullConfig,
    NoiseConfig,
    CavityConfig,
    SimulationConfig,
    DetectionConfig,
    SpectrumConfig,
    clone_config,
    run_simulation_with_upper_branch,
    compute_psd,
    save_results_npz,
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _band_power(freqs_mhz: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Integrated PSD over a frequency band in MHz."""
    mask = (freqs_mhz >= fmin) & (freqs_mhz <= fmax)
    if not np.any(mask):
        return np.nan
    return float(np.trapz(psd[mask], freqs_mhz[mask]))


def _variance_after_mean_removal(x: np.ndarray) -> float:
    """Variance of the fluctuating part only."""
    x = np.asarray(x)
    return float(np.var(x - np.mean(x)))


def _linear_fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 1e-10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ridge regression with feature standardization.

    Returns
    -------
    beta : ndarray
        Regression coefficients including intercept in standardized space.
    mean : ndarray
        Feature means.
    std : ndarray
        Feature standard deviations.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0

    Xs = (X - mean) / std
    A = np.column_stack([np.ones(Xs.shape[0]), Xs])

    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0.0  # do not regularize intercept
    beta = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return beta, mean, std


def _linear_predict(X: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Xs = (X - mean) / std
    A = np.column_stack([np.ones(Xs.shape[0]), Xs])
    return A @ beta


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    span = float(np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / span) if span > 0 else np.nan
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else np.nan
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse, "corr": corr}


# ---------------------------------------------------------------------------
# Physics/simulation config
# ---------------------------------------------------------------------------

def make_base_config(quick: bool = False) -> FullConfig:
    """
    Baseline config inherited from Polariton_Microcavity_bistability.py.

    Important choices:
    - amplitude-noise only;
    - no extra detector shot/electronic noise by default, so the learning first
      tests the cavity transfer itself;
    - balanced_sum detection by default, but the feature extractor also uses
      saved field quadratures x_out / p_out.
    """
    cfg = FullConfig()

    cfg.noise.mode = "amplitude"
    cfg.noise.strength_phase = 0.0
    cfg.noise.cutoff_mhz = 2000.0

    cfg.detection.mode = "balanced_sum"
    cfg.detection.add_shot_noise = False
    cfg.detection.electronic_noise_psd_per_mhz = 0.0
    cfg.detection.simulate_vacuum_port = False

    # Keep these moderate for dataset generation. Increase duration_ps for
    # cleaner PSD features once the pipeline is validated.
    if quick:
        cfg.sim.duration_ps = 1.0e5
        cfg.sim.dt_ps = 2.0
        cfg.sim.store_every = 50
        cfg.spectrum.nperseg = 2**9
    else:
        cfg.sim.duration_ps = 5.0e5
        cfg.sim.dt_ps = 1.0
        cfg.sim.store_every = 100
        cfg.spectrum.nperseg = 2**11

    cfg.sim.discard_fraction = 0.2
    cfg.sim.integrator = "rk4"

    return cfg


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "var_x_out_excess",
    "var_p_out_excess",
    "cov_xp_out_excess",
    "var_i_det_excess",
    "band_x_out_0_500MHz",
    "band_x_out_500_2000MHz",
    "band_p_out_0_500MHz",
    "band_p_out_500_2000MHz",
]


def extract_features(res: Dict[str, np.ndarray], baseline: Dict[str, float] | None = None) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract low-dimensional output-noise features.

    baseline is measured at sigma_amp = 0 and subtracted from variance-like
    quantities. This makes the fit focus on the added input noise response.
    """
    x_out = res["x_out"]
    p_out = res["p_out"]
    i_det = res["i_det_t"]

    var_x = _variance_after_mean_removal(x_out)
    var_p = _variance_after_mean_removal(p_out)
    cov_xp = float(np.cov(x_out - np.mean(x_out), p_out - np.mean(p_out), ddof=0)[0, 1])
    var_i = _variance_after_mean_removal(i_det)

    fs_store_mhz = float(np.ravel(res["fs_store_mhz"])[0])
    f_x, psd_x = compute_psd(x_out - np.mean(x_out), fs_store_mhz, SpectrumConfig(nperseg=min(2**11, len(x_out))))
    f_p, psd_p = compute_psd(p_out - np.mean(p_out), fs_store_mhz, SpectrumConfig(nperseg=min(2**11, len(p_out))))

    vals = {
        "var_x_out": var_x,
        "var_p_out": var_p,
        "cov_xp_out": cov_xp,
        "var_i_det": var_i,
        "band_x_out_0_500MHz": _band_power(f_x, psd_x, 0.0, 500.0),
        "band_x_out_500_2000MHz": _band_power(f_x, psd_x, 500.0, 2000.0),
        "band_p_out_0_500MHz": _band_power(f_p, psd_p, 0.0, 500.0),
        "band_p_out_500_2000MHz": _band_power(f_p, psd_p, 500.0, 2000.0),
    }

    if baseline is None:
        baseline = {}

    feat_dict = {
        "var_x_out_excess": vals["var_x_out"] - baseline.get("var_x_out", 0.0),
        "var_p_out_excess": vals["var_p_out"] - baseline.get("var_p_out", 0.0),
        "cov_xp_out_excess": vals["cov_xp_out"] - baseline.get("cov_xp_out", 0.0),
        "var_i_det_excess": vals["var_i_det"] - baseline.get("var_i_det", 0.0),
        "band_x_out_0_500MHz": vals["band_x_out_0_500MHz"] - baseline.get("band_x_out_0_500MHz", 0.0),
        "band_x_out_500_2000MHz": vals["band_x_out_500_2000MHz"] - baseline.get("band_x_out_500_2000MHz", 0.0),
        "band_p_out_0_500MHz": vals["band_p_out_0_500MHz"] - baseline.get("band_p_out_0_500MHz", 0.0),
        "band_p_out_500_2000MHz": vals["band_p_out_500_2000MHz"] - baseline.get("band_p_out_500_2000MHz", 0.0),
    }

    x = np.array([feat_dict[name] for name in FEATURE_NAMES], dtype=float)
    return x, {**vals, **feat_dict}


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def simulate_one(
    base_cfg: FullConfig,
    sigma_amp: float,
    seed: int,
    F_low: complex,
    F_high: complex,
    F_work: complex,
) -> Dict[str, np.ndarray]:
    cfg = clone_config(base_cfg)
    cfg.noise.mode = "amplitude"
    cfg.noise.strength_amp = float(sigma_amp)
    cfg.noise.strength_phase = 0.0
    cfg.noise.seed = int(seed)
    cfg.cavity.F_s = F_work
    return run_simulation_with_upper_branch(cfg, F_low=F_low, F_high=F_high, F_work=F_work)


def build_dataset(
    base_cfg: FullConfig,
    sigmas: Iterable[float],
    repeats: int,
    seed0: int,
    F_low: complex,
    F_high: complex,
    F_work: complex,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, list[dict]]:
    """
    Build a dataset with repeated noise realizations for each sigma.
    """
    # Baseline at sigma=0 for excess-noise features.
    baseline_res = simulate_one(base_cfg, sigma_amp=0.0, seed=seed0 - 1, F_low=F_low, F_high=F_high, F_work=F_work)
    _, baseline_vals = extract_features(baseline_res, baseline=None)

    X_rows = []
    y_rows = []
    rows = []

    job = 0
    for sigma in sigmas:
        for rep in range(repeats):
            seed = seed0 + job
            print(f"[{label}] sigma_amp={sigma:.6g}, repeat={rep + 1}/{repeats}, seed={seed}")
            res = simulate_one(base_cfg, sigma_amp=float(sigma), seed=seed, F_low=F_low, F_high=F_high, F_work=F_work)
            x, vals = extract_features(res, baseline=baseline_vals)

            X_rows.append(x)
            y_rows.append(float(sigma))
            rows.append({
                "label": label,
                "sigma_amp": float(sigma),
                "repeat": rep,
                "seed": seed,
                **vals,
            })
            job += 1

    return np.vstack(X_rows), np.array(y_rows), rows


# ---------------------------------------------------------------------------
# Plotting / saving
# ---------------------------------------------------------------------------

def plot_learning_results(
    y_train: np.ndarray,
    pred_train: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    outdir: Path,
    show: bool = False,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(y_train, pred_train, label="train")
    plt.scatter(y_test, pred_test, label="test")
    lo = min(np.min(y_train), np.min(y_test), np.min(pred_train), np.min(pred_test))
    hi = max(np.max(y_train), np.max(y_test), np.max(pred_train), np.max(pred_test))
    plt.plot([lo, hi], [lo, hi], "--", label="ideal")
    plt.xlabel(r"Injected amplitude noise $\sigma_\mathrm{amp}$")
    plt.ylabel(r"Predicted $\sigma_\mathrm{amp}$")
    plt.title("Learning input amplitude noise from output features")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "predicted_vs_true_sigma_amp.png", dpi=180)

    fig = plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred_test - y_test)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel(r"Injected amplitude noise $\sigma_\mathrm{amp}$")
    plt.ylabel("Prediction error")
    plt.title("Test residuals")
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "test_residuals_sigma_amp.png", dpi=180)

    if show:
        plt.show()
    else:
        plt.close("all")


def save_learning_npz(
    path: Path,
    base_cfg: FullConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    pred_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pred_test: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    train_rows: list[dict],
    test_rows: list[dict],
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    metadata = {
        "feature_names": FEATURE_NAMES,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "base_cfg": {
            "noise": asdict(base_cfg.noise),
            "cavity": asdict(base_cfg.cavity),
            "sim": asdict(base_cfg.sim),
            "detection": asdict(base_cfg.detection),
            "spectrum": asdict(base_cfg.spectrum),
        },
        "train_rows": train_rows,
        "test_rows": test_rows,
    }

    def json_safe(obj):
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        return obj

    np.savez_compressed(
        path,
        metadata_json=json.dumps(json_safe(metadata), indent=2),
        feature_names=np.array(FEATURE_NAMES),
        X_train=X_train,
        y_train=y_train,
        pred_train=pred_train,
        X_test=X_test,
        y_test=y_test,
        pred_test=pred_test,
        beta=beta,
        feature_mean=mean,
        feature_std=std,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a smaller/faster simulation.")
    parser.add_argument("--n-train", type=int, default=12)
    parser.add_argument("--n-test", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--sigma-max", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--alpha", type=float, default=1e-10)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", type=str, default="learning_bistability_results.npz")
    parser.add_argument("--plot-dir", type=str, default="Plots_learning_bistability")
    args = parser.parse_args()

    base_cfg = make_base_config(quick=args.quick)

    # Same upper-branch preparation values as your current bistability script.
    F_low = 0.3 + 0j
    F_high = 1.2 + 0j
    F_work = 0.75 + 0j

    # Keep train/test sigmas distinct, so this tests interpolation rather than
    # exactly repeating the same sigma values.
    sigmas_train = np.linspace(args.sigma_min, args.sigma_max, args.n_train)
    if args.n_test > 1:
        step = (args.sigma_max - args.sigma_min) / max(args.n_train - 1, 1)
        sigmas_test = np.linspace(args.sigma_min + 0.5 * step, args.sigma_max - 0.5 * step, args.n_test)
    else:
        sigmas_test = np.array([0.5 * (args.sigma_min + args.sigma_max)])

    X_train, y_train, train_rows = build_dataset(
        base_cfg=base_cfg,
        sigmas=sigmas_train,
        repeats=args.repeats,
        seed0=args.seed,
        F_low=F_low,
        F_high=F_high,
        F_work=F_work,
        label="train",
    )

    X_test, y_test, test_rows = build_dataset(
        base_cfg=base_cfg,
        sigmas=sigmas_test,
        repeats=args.repeats,
        seed0=args.seed + 100000,
        F_low=F_low,
        F_high=F_high,
        F_work=F_work,
        label="test",
    )

    beta, mean, std = _linear_fit_ridge(X_train, y_train, alpha=args.alpha)
    pred_train = _linear_predict(X_train, beta, mean, std)
    pred_test = _linear_predict(X_test, beta, mean, std)

    train_metrics = _metrics(y_train, pred_train)
    test_metrics = _metrics(y_test, pred_test)

    print("\n=== Feature names ===")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{i:2d}: {name}")

    print("\n=== Train metrics ===")
    for k, v in train_metrics.items():
        print(f"{k:>8s}: {v:.6g}")

    print("\n=== Test metrics ===")
    for k, v in test_metrics.items():
        print(f"{k:>8s}: {v:.6g}")

    plot_learning_results(
        y_train=y_train,
        pred_train=pred_train,
        y_test=y_test,
        pred_test=pred_test,
        outdir=Path(args.plot_dir),
        show=args.show,
    )

    save_learning_npz(
        path=Path(args.out),
        base_cfg=base_cfg,
        X_train=X_train,
        y_train=y_train,
        pred_train=pred_train,
        X_test=X_test,
        y_test=y_test,
        pred_test=pred_test,
        beta=beta,
        mean=mean,
        std=std,
        train_rows=train_rows,
        test_rows=test_rows,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    print(f"\nSaved learning dataset and model to: {args.out}")
    print(f"Saved figures to: {args.plot_dir}/")


if __name__ == "__main__":
    main()
