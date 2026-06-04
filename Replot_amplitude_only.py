#!/usr/bin/env python3
"""
Replot saved amplitude-only reconstruction results without recomputing datasets/GPE.

Expected folder structure, for example:

output/amplitude_only_reconstruction/
    summary.json
    amplitude/
        amplitude_only_reconstruction_amplitude.npz
    phase/
        amplitude_only_reconstruction_phase.npz
    both/
        amplitude_only_reconstruction_both.npz

Usage:
    python Replot_amplitude_only.py --indir output/amplitude_only_reconstruction
    python Replot_amplitude_only.py --indir output/amplitude_only_reconstruction --show
    python Replot_amplitude_only.py --indir output/amplitude_only_reconstruction --cases amplitude phase both
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_TARGET_NAMES = ["Var_X_in", "Var_P_in", "Cov_XP_in"]
DEFAULT_BASE_FEATURE_NAMES = ["mean", "var", "total_power"]
DEFAULT_EXTRA_FEATURE_NAMES = ["std", "centroid", "log_mean", "log_std"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replot saved amplitude-only reconstruction outputs."
    )
    p.add_argument(
        "--indir",
        type=Path,
        required=True,
        help="Directory containing summary.json and case subfolders.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Where to save replotted figures. Default: <indir>/replots.",
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help="Cases to replot. Default: infer from summary.json or subfolders.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Keep matplotlib windows open at the end.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI for saved PNGs.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, only plot the top-k feature importances per target.",
    )
    p.add_argument(
        "--also-pdf",
        action="store_true",
        help="Also save figures as PDF.",
    )
    return p.parse_args()


def load_summary(indir: Path) -> Dict:
    path = indir / "summary.json"
    if not path.exists():
        print(f"[warning] No summary.json found at {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def infer_cases(indir: Path, summary: Dict, cases_arg: Optional[Sequence[str]]) -> List[str]:
    if cases_arg:
        return list(cases_arg)

    if "summary" in summary and isinstance(summary["summary"], dict):
        return list(summary["summary"].keys())

    cases = []
    for p in indir.iterdir():
        if p.is_dir():
            npz = p / f"amplitude_only_reconstruction_{p.name}.npz"
            if npz.exists():
                cases.append(p.name)
    return sorted(cases)


def find_npz(indir: Path, case: str) -> Path:
    candidates = [
        indir / case / f"amplitude_only_reconstruction_{case}.npz",
        indir / f"amplitude_only_reconstruction_{case}.npz",
    ]
    for c in candidates:
        if c.exists():
            return c

    matches = list(indir.rglob(f"*{case}*.npz"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not find .npz file for case={case} in {indir}")


def as_str_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [str(v) for v in x.tolist()]
    return [str(v) for v in x]


def default_feature_names(n_features: int) -> List[str]:
    n_bands = n_features - len(DEFAULT_BASE_FEATURE_NAMES) - len(DEFAULT_EXTRA_FEATURE_NAMES)
    if n_bands < 0:
        return [f"feature_{i}" for i in range(n_features)]
    return (
        DEFAULT_BASE_FEATURE_NAMES
        + [f"band_{i}" for i in range(n_bands)]
        + DEFAULT_EXTRA_FEATURE_NAMES
    )


def get_feature_names(npz_data, summary_case: Optional[Dict], n_features: int) -> List[str]:
    if "feature_names" in npz_data.files:
        names = as_str_list(npz_data["feature_names"])
        if len(names) == n_features:
            return names

    if summary_case and "feature_names" in summary_case:
        names = as_str_list(summary_case["feature_names"])
        if len(names) == n_features:
            return names

    return default_feature_names(n_features)


def get_target_names(npz_data, summary_case: Optional[Dict], n_targets: int) -> List[str]:
    if "target_names" in npz_data.files:
        names = as_str_list(npz_data["target_names"])
        if len(names) == n_targets:
            return names

    if summary_case and "target_names" in summary_case:
        names = as_str_list(summary_case["target_names"])
        if len(names) == n_targets:
            return names

    if n_targets == 3:
        return DEFAULT_TARGET_NAMES
    return [f"target_{i}" for i in range(n_targets)]


def save_fig(fig, outpath: Path, dpi: int = 300, also_pdf: bool = False) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    print(f"Saved {outpath.resolve()}")
    if also_pdf:
        pdf_path = outpath.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved {pdf_path.resolve()}")


def r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-30)


def nrmse_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.sqrt(np.mean(y_true ** 2)) + 1e-30)) * 100.0


def plot_predictions(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    target_names: Sequence[str],
    case: str,
) -> plt.Figure:
    n_targets = Y_test.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(4.3 * n_targets, 4.0))
    if n_targets == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        yt = Y_test[:, j]
        yp = Y_pred[:, j]
        r2 = r2_score_1d(yt, yp)
        nrmse = nrmse_1d(yt, yp)

        ax.scatter(yt, yp, s=35, alpha=0.8)
        mn = min(float(np.min(yt)), float(np.min(yp)))
        mx = max(float(np.max(yt)), float(np.max(yp)))
        if np.isclose(mn, mx):
            pad = 1.0 if mn == 0 else 0.05 * abs(mn)
            mn -= pad
            mx += pad
        ax.plot([mn, mx], [mn, mx], "--", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{target_names[j]}\nR²={r2:.3f}, NRMSE={nrmse:.2e}")
        ax.grid(False, alpha=0.3)

    fig.suptitle(f"Predictions | case={case}")
    fig.tight_layout()
    return fig


def plot_residuals(
    Y_test: np.ndarray,
    Y_pred: np.ndarray,
    target_names: Sequence[str],
    case: str,
) -> plt.Figure:
    n_targets = Y_test.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(4.3 * n_targets, 4.0))
    if n_targets == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        res = Y_pred[:, j] - Y_test[:, j]
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.scatter(Y_test[:, j], res, s=35, alpha=0.8)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted - true")
        ax.set_title(target_names[j])
        ax.grid(False, alpha=0.3)

    fig.suptitle(f"Residuals | case={case}")
    fig.tight_layout()
    return fig


def coef_from_summary_labeled(
    summary_case: Optional[Dict],
    target_names: Sequence[str],
    feature_names: Sequence[str],
) -> Optional[np.ndarray]:
    if not summary_case:
        return None

    labeled = summary_case.get("ridge_coef_labeled", None)
    if not isinstance(labeled, dict):
        return None

    rows = []
    for target in target_names:
        row_dict = labeled.get(str(target), None)
        if not isinstance(row_dict, dict):
            return None
        rows.append([float(row_dict.get(str(feat), 0.0)) for feat in feature_names])
    return np.asarray(rows, dtype=float)


def get_ridge_coef(
    npz_data,
    summary_case: Optional[Dict],
    target_names: Sequence[str],
    feature_names: Sequence[str],
) -> Optional[np.ndarray]:
    if "ridge_coef" in npz_data.files:
        coef = np.asarray(npz_data["ridge_coef"], dtype=float)
        # Expected shape: n_targets x n_features.
        if coef.ndim == 2:
            return coef

    if summary_case and "ridge_coef" in summary_case:
        coef = np.asarray(summary_case["ridge_coef"], dtype=float)
        if coef.ndim == 2:
            return coef

    return coef_from_summary_labeled(summary_case, target_names, feature_names)


def plot_feature_importance(
    coef: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    case: str,
    top_k: Optional[int] = None,
) -> List[Tuple[str, plt.Figure]]:
    figs = []

    for target, coef_row in zip(target_names, coef):
        abs_coef = np.abs(coef_row)
        importance = abs_coef / (abs_coef.sum() + 1e-30)

        names = np.asarray(feature_names)
        values = np.asarray(importance)

        if top_k is not None and top_k > 0 and top_k < len(values):
            idx = np.argsort(values)[-top_k:][::-1]
            names = names[idx]
            values = values[idx]
        else:
            idx = np.argsort(values)[::-1]
            names = names[idx]
            values = values[idx]

        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(values)), 4.0))
        ax.bar(names, values)
        ax.set_ylabel("Normalized |ridge coefficient|")
        ax.set_xlabel("Feature")
        ax.set_title(f"Feature importance for {target} | case={case}")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(False, axis="y", alpha=0.3)
        fig.tight_layout()
        figs.append((str(target), fig))

    return figs


def plot_feature_correlation_matrix(
    X: np.ndarray,
    feature_names: Sequence[str],
    case: str,
) -> plt.Figure:

    n_features = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr)

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90)
    ax.set_yticklabels(feature_names)
    ax.set_title(f"Feature correlation matrix | case={case}")

    for i in range(n_features):
        for j in range(n_features):
            ax.text(
                j,
                i,
                f"{corr[i,j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    return fig


def plot_feature_target_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    feature_names: Sequence[str],
    target_names: Sequence[str],
    case: str = "",
) -> Tuple[plt.Figure, np.ndarray]:

    n_features = X.shape[1]
    n_targets = Y.shape[1]

    corr = np.zeros((n_features, n_targets))

    for i in range(n_features):
        for j in range(n_targets):
            xi = X[:, i]
            yj = Y[:, j]
            if np.std(xi) < 1e-12 or np.std(yj) < 1e-12:
                corr[i, j] = 0.0
            else:
                corr[i, j] = np.corrcoef(xi, yj)[0, 1]

    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title(f"Feature ↔ Target correlations | case={case}")

    for i in range(n_features):
        for j in range(n_targets):
            ax.text(
                j,
                i,
                f"{corr[i,j]:.2f}",
                ha="center",
                va="center",
            )
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()

    return fig


def plot_gain_scan_if_available(npz_data, case: str) -> Optional[plt.Figure]:
    keys = npz_data.files
    needed = ["test_gain_amp_db", "test_gain_phase_db", "Y_test", "Y_pred"]
    if not all(k in keys for k in needed):
        return None

    Y_test = npz_data["Y_test"]
    Y_pred = npz_data["Y_pred"]
    err = np.sqrt(np.mean((Y_pred - Y_test) ** 2, axis=1))

    gain_amp = npz_data["test_gain_amp_db"]
    gain_phase = npz_data["test_gain_phase_db"]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    if case == "phase":
        x = gain_phase
        xlabel = "Phase gain (dB)"
    elif case == "amplitude":
        x = gain_amp
        xlabel = "Amplitude gain (dB)"
    else:
        # For both, use color as phase gain and x as amplitude gain.
        sc = ax.scatter(gain_amp, gain_phase, c=err, s=45, alpha=0.85)
        fig.colorbar(sc, ax=ax, label="NRMSE $\epsilon$")
        ax.set_xlabel("Amplitude gain (dB)")
        ax.set_ylabel("Phase gain (dB)")
        ax.set_title(f"Prediction error vs gains | case={case}")
        ax.grid(False, alpha=0.3)
        fig.tight_layout()
        return fig

    ax.scatter(x, err, s=45, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NRMSE $\epsilon$")
    ax.set_title(f"Prediction error vs gain | case={case}")
    ax.grid(False, alpha=0.3)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    indir = args.indir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve() if args.outdir else indir / "replots"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(indir)
    summary_cases = summary.get("summary", {}) if isinstance(summary, dict) else {}
    cases = infer_cases(indir, summary, args.cases)

    if not cases:
        raise RuntimeError(
            "No cases found. Pass --cases amplitude phase both or check --indir."
        )

    print("Input directory:", indir)
    print("Output directory:", outdir)
    print("Cases:", cases)

    for case in cases:
        print("\n" + "=" * 80)
        print(f"Replotting case={case}")
        print("=" * 80)

        npz_path = find_npz(indir, case)
        print("Loading:", npz_path)
        data = np.load(npz_path, allow_pickle=True)
        summary_case = summary_cases.get(case, None)

        Y_test = np.asarray(data["Y_test"], dtype=float)
        Y_pred = np.asarray(data["Y_pred"], dtype=float)
        X_test = np.asarray(data["X_test"], dtype=float) if "X_test" in data.files else None

        target_names = get_target_names(data, summary_case, Y_test.shape[1])
        n_features = X_test.shape[1] if X_test is not None else None

        if n_features is None:
            coef_tmp = get_ridge_coef(data, summary_case, target_names, [])
            if coef_tmp is not None:
                n_features = coef_tmp.shape[1]
            else:
                n_features = 0

        feature_names = get_feature_names(data, summary_case, n_features)

        fig = plot_predictions(Y_test, Y_pred, target_names, case)
        save_fig(fig, outdir / case / f"predictions_{case}.png", args.dpi, args.also_pdf)

        fig = plot_residuals(Y_test, Y_pred, target_names, case)
        save_fig(fig, outdir / case / f"residuals_{case}.png", args.dpi, args.also_pdf)

        coef = get_ridge_coef(data, summary_case, target_names, feature_names)
        if coef is not None:
            figs = plot_feature_importance(
                coef, feature_names, target_names, case, top_k=args.top_k
            )
            for target, fig in figs:
                safe_target = target.replace("/", "_").replace(" ", "_")
                save_fig(
                    fig,
                    outdir / case / f"importance_{case}_{safe_target}.png",
                    args.dpi,
                    args.also_pdf,
                )
        else:
            print("[warning] No ridge coefficients found; skipping importance plots.")

        if X_test is not None and len(feature_names) == X_test.shape[1]:
            fig = plot_feature_correlation_matrix(X_test, feature_names, case)
            save_fig(
                fig,
                outdir / case / f"feature_correlation_{case}.png",
                args.dpi,
                args.also_pdf,
            )

            fig = plot_feature_target_correlations(X_test, Y_test, feature_names, target_names, case)
            save_fig(
                fig,
                outdir / case / f"feature_target_correlation_{case}.png",
                args.dpi,
                args.also_pdf,
            )

        fig = plot_gain_scan_if_available(data, case)
        if fig is not None:
            save_fig(fig, outdir / case / f"error_vs_gain_{case}.png", args.dpi, args.also_pdf)

    if args.show:
        plt.show(block=True)
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
