from __future__ import annotations

"""
run_bistability.py

Run one polariton microcavity bistability simulation and save the .npz result.

Usage:
    python run_bistability.py

Fast test:
    python run_bistability.py --quick --show

Amplitude noise only:
    python run_bistability.py --noise-mode amplitude --sigma-amp 0.05 --sigma-phase 0
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src.polariton_bistability_core import (
    FullConfig,
    PumpProtocol,
    BistabilityExperiment,
    ResultPlotter,
    estimate_quadrature_variances,
)


def make_config(args: argparse.Namespace) -> FullConfig:
    cfg = FullConfig()

    cfg.noise.mode = args.noise_mode
    cfg.noise.strength_amp = args.sigma_amp
    cfg.noise.strength_phase = args.sigma_phase
    cfg.noise.cutoff_mhz = args.cutoff_mhz
    cfg.noise.seed = args.seed

    cfg.detection.mode = args.detection_mode
    cfg.detection.simulate_vacuum_port = not args.no_vacuum_port
    cfg.detection.add_shot_noise = args.add_shot_noise
    cfg.detection.electronic_noise_psd_per_mhz = args.electronic_noise

    if args.quick:
        cfg.sim.duration_ps = 1.0e5
        cfg.sim.dt_ps = 2.0
        cfg.sim.store_every = 50
        cfg.spectrum.nperseg = 2**9
    else:
        cfg.sim.duration_ps = args.duration_ps
        cfg.sim.dt_ps = args.dt_ps
        cfg.sim.store_every = args.store_every
        cfg.spectrum.nperseg = args.nperseg

    cfg.sim.discard_fraction = args.discard_fraction
    cfg.sim.integrator = args.integrator

    return cfg


def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--out", type=str, default="Results/polariton_bistability_results.npz")
    p.add_argument("--show", action="store_true")
    p.add_argument("--quick", action="store_true")

    p.add_argument("--noise-mode", choices=["amplitude", "phase", "both"], default="both")
    p.add_argument("--sigma-amp", type=float, default=0.02 * 10 ** (5.0 / 20.0))
    p.add_argument("--sigma-phase", type=float, default=0.02 * 10 ** (5.0 / 20.0))
    p.add_argument("--cutoff-mhz", type=float, default=2000.0)
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument("--detection-mode", choices=["balanced_sum", "balanced_diff"], default="balanced_sum")
    p.add_argument("--no-vacuum-port", action="store_true")
    p.add_argument("--add-shot-noise", action="store_true")
    p.add_argument("--electronic-noise", type=float, default=0.0)

    p.add_argument("--duration-ps", type=float, default=1.0e6)
    p.add_argument("--dt-ps", type=float, default=1.0)
    p.add_argument("--discard-fraction", type=float, default=0.1)
    p.add_argument("--store-every", type=int, default=250)
    p.add_argument("--integrator", choices=["rk4", "heun", "euler"], default="rk4")
    p.add_argument("--nperseg", type=int, default=2**10)

    p.add_argument("--F-low", type=float, default=0.3)
    p.add_argument("--F-high", type=float, default=1.2)
    p.add_argument("--F-work", type=float, default=0.75)
    p.add_argument("--no-bistability", action="store_true")

    args = p.parse_args()

    cfg = make_config(args)
    pump = PumpProtocol(
        F_low=args.F_low + 0j,
        F_high=args.F_high + 0j,
        F_work=args.F_work + 0j,
    )

    exp = BistabilityExperiment(cfg, pump)
    results = exp.run_full(include_bistability=not args.no_bistability)

    print("\nInput drive quadrature stats:")
    for k, v in estimate_quadrature_variances(results["F_t"]).items():
        print(f"  {k:>10s} = {v:.6g}")

    print("\nOutput field quadrature stats:")
    for k, v in estimate_quadrature_variances(results["s_out_t"]).items():
        print(f"  {k:>10s} = {v:.6g}")

    out = Path(args.out)
    results.save_npz(out, cfg)
    print(f"\nSaved results to {out}")

    if args.show:
        plotter = ResultPlotter(results)
        plotter.bistability()
        plotter.balanced_psds()
        plt.show()


if __name__ == "__main__":
    main()
