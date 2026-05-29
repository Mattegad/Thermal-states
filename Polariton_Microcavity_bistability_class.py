from __future__ import annotations

"""
Class-based refactor of Polariton_Microcavity_bistability2.py.

This file keeps the same physics, same config dataclasses, and same saved-array
names as the procedural version, so Replot_bistability.py can still read the
produced .npz files.

Main objects
------------
NoiseGenerator
    Creates amplitude/phase/both band-limited drive noise.

KerrCavity
    Integrates the single-mode polariton Gross-Pitaevskii / Kerr equation and
    applies the input-output relation.

BalancedDetector
    Builds the 50/50 balanced detection currents, including the optional vacuum
    port convention already used in the original script.

SpectrumAnalyzer
    Computes Welch PSDs and optional RBW averaging.

BistabilityExperiment
    High-level object that prepares the upper branch, runs the noisy dynamics,
    computes diagnostics/PSDs, and returns a results dict compatible with your
    old replot script.

Compatibility wrappers are kept at the bottom:
    run_simulation_with_upper_branch(...)
    compute_bistability_curve(...)
    prepare_upper_branch(...)
    save_results_npz(...)
so old scripts can progressively migrate to the class API.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple
import copy
import json

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

try:
    from scipy.signal import welch
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
NoiseMode = Literal["amplitude", "phase", "both"]
DetectionMode = Literal["homodyne", "balanced_sum", "balanced_diff"]
ShotNoiseMode = Literal["none", "fixed", "photocurrent"]
IntegratorName = Literal["rk4", "heun", "euler"]

HBAR_MEV_PS = 0.6582119514
MHZ_PER_INV_PS = 1.0e6
INV_PS_PER_MHZ = 1.0 / MHZ_PER_INV_PS


# -----------------------------------------------------------------------------
# Config containers
# -----------------------------------------------------------------------------

@dataclass
class CavityConfig:
    detuning_inv_ps: float = 1.4e-1 / HBAR_MEV_PS
    nonlinearity_inv_ps: float = 1.2e-2 / HBAR_MEV_PS
    loss_inv_ps: float = 7e-2 / HBAR_MEV_PS
    kappa_out_inv_ps: float = 7e-2 / HBAR_MEV_PS
    F_s: complex = 0.7 + 0j
    psi0: complex = 0.0 + 0.0j


@dataclass
class SimulationConfig:
    duration_ps: float = 1.0e6
    dt_ps: float = 1.0
    discard_fraction: float = 0.1
    integrator: IntegratorName = "rk4"
    store_every: int = 250


@dataclass
class DetectionConfig:
    mode: DetectionMode = "balanced_sum"

    lo_amplitude: float = 50.0
    lo_phase_rad: float = np.pi / 2

    detection_efficiency: float = 1.0
    responsivity: float = 1.0

    add_shot_noise: bool = True
    shot_noise_mode: ShotNoiseMode = "photocurrent"
    shot_noise_psd_per_mhz: float = 0.0
    shot_noise_gain_per_current: float = 1.0
    shot_noise_use_instantaneous_photocurrent: bool = False
    electronic_noise_psd_per_mhz: float = 0.0

    simulate_vacuum_port: bool = True
    sigma_vac: float = 0.02


@dataclass
class NoiseConfig:
    mode: NoiseMode = "both"
    cutoff_mhz: float = 2000.0
    gain_dB: float = 5.0
    strength_amp: float = DetectionConfig.sigma_vac * 10 ** (gain_dB / 20)
    strength_phase: float = DetectionConfig.sigma_vac * 10 ** (gain_dB / 20)
    seed: int = 12345


@dataclass
class SpectrumConfig:
    nperseg: int = 2**10
    window: str = "hann"
    detrend: str = "constant"
    average: str = "mean"


@dataclass
class FullConfig:
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    cavity: CavityConfig = field(default_factory=CavityConfig)
    sim: SimulationConfig = field(default_factory=SimulationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)


@dataclass
class PumpProtocol:
    F_low: complex = 0.3 + 0j
    F_high: complex = 1.2 + 0j
    F_work: complex = 0.75 + 0j
    rise_fraction: float = 0.2
    fall_fraction: float = 0.7


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def clone_config(base_cfg: FullConfig) -> FullConfig:
    return copy.deepcopy(base_cfg)


def time_axis(duration_ps: float, dt_ps: float) -> Array:
    n = int(np.round(duration_ps / dt_ps))
    return np.arange(n, dtype=np.float64) * dt_ps


def frequency_axis(n: int, dt_ps: float) -> Array:
    return np.fft.fftfreq(n, d=dt_ps) * MHZ_PER_INV_PS


def complex_to_quadratures(z: ComplexArray) -> Tuple[Array, Array]:
    return np.sqrt(2.0) * np.real(z), np.sqrt(2.0) * np.imag(z)


def quadrature_projection(z: ComplexArray, theta: float) -> Array:
    return np.sqrt(2.0) * np.real(np.exp(-1j * theta) * z)


def estimate_quadrature_variances(z: ComplexArray) -> Dict[str, float]:
    x, p = complex_to_quadratures(z)
    return {
        "var_x": float(np.var(x)),
        "var_p": float(np.var(p)),
        "cov_xp": float(np.cov(x, p, ddof=0)[0, 1]),
        "mean_x": float(np.mean(x)),
        "mean_p": float(np.mean(p)),
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, complex):
        return {"__complex__": True, "real": float(np.real(obj)), "imag": float(np.imag(obj))}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_results_npz(path: str | Path, cfg: FullConfig, results: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "noise": asdict(cfg.noise),
        "cavity": asdict(cfg.cavity),
        "sim": asdict(cfg.sim),
        "detection": asdict(cfg.detection),
        "spectrum": asdict(cfg.spectrum),
    }
    np.savez_compressed(path, metadata_json=json.dumps(_json_safe(meta), indent=2), **results)


# -----------------------------------------------------------------------------
# Components
# -----------------------------------------------------------------------------

class NoiseGenerator:
    def __init__(self, cfg: NoiseConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    @staticmethod
    def band_limited_real_gaussian(
        n: int,
        dt_ps: float,
        cutoff_mhz: float,
        rms: float,
        rng: np.random.Generator,
    ) -> Array:
        if rms == 0.0:
            return np.zeros(n, dtype=np.float64)

        freqs = np.fft.rfftfreq(n, d=dt_ps) * MHZ_PER_INV_PS
        coeff = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
        coeff *= (freqs <= cutoff_mhz).astype(np.float64)
        coeff[0] = coeff[0].real + 0j
        if n % 2 == 0:
            coeff[-1] = coeff[-1].real + 0j

        x = np.fft.irfft(coeff, n=n)
        x -= np.mean(x)
        std = np.std(x)
        if std > 0:
            x *= rms / std
        return x.astype(np.float64)

    def drive_noise(self, t: Array) -> Tuple[ComplexArray, Dict[str, Array]]:
        n = t.size
        dt_ps = float(t[1] - t[0]) if n > 1 else 1.0

        amp = np.zeros(n, dtype=np.float64)
        phase = np.zeros(n, dtype=np.float64)

        if self.cfg.mode in ("amplitude", "both") and self.cfg.strength_amp != 0.0:
            amp = self.band_limited_real_gaussian(n, dt_ps, self.cfg.cutoff_mhz, self.cfg.strength_amp, self.rng)
        if self.cfg.mode in ("phase", "both") and self.cfg.strength_phase != 0.0:
            phase = self.band_limited_real_gaussian(n, dt_ps, self.cfg.cutoff_mhz, self.cfg.strength_phase, self.rng)

        F_n = (amp + 1j * phase) / np.sqrt(2.0)
        return F_n.astype(np.complex128), {
            "amp_noise": amp,
            "phase_noise": phase,
            "cutoff_mhz": np.array([self.cfg.cutoff_mhz], dtype=np.float64),
        }


class KerrCavity:
    def __init__(self, cfg: CavityConfig, integrator: IntegratorName = "rk4") -> None:
        self.cfg = cfg
        self.integrator = integrator

    def rhs(self, psi: complex, F: complex) -> complex:
        c = self.cfg
        return (
            1j * c.detuning_inv_ps * psi
            - 1j * c.nonlinearity_inv_ps * (abs(psi) ** 2) * psi
            - 0.5 * c.loss_inv_ps * psi
            + np.sqrt(c.kappa_out_inv_ps) * F
        )

    def integrate(self, t: Array, F_t: ComplexArray, psi0: Optional[complex] = None) -> ComplexArray:
        n = t.size
        psi = np.empty(n, dtype=np.complex128)
        psi[0] = self.cfg.psi0 if psi0 is None else psi0
        dt = float(t[1] - t[0]) if n > 1 else 1.0

        for k in range(n - 1):
            y = psi[k]
            f0 = F_t[k]
            f1 = F_t[k + 1]
            fmid = 0.5 * (f0 + f1)

            if self.integrator == "euler":
                psi[k + 1] = y + dt * self.rhs(y, f0)
            elif self.integrator == "heun":
                y_pred = y + dt * self.rhs(y, f0)
                psi[k + 1] = y + 0.5 * dt * (self.rhs(y, f0) + self.rhs(y_pred, f1))
            elif self.integrator == "rk4":
                k1 = self.rhs(y, f0)
                k2 = self.rhs(y + 0.5 * dt * k1, fmid)
                k3 = self.rhs(y + 0.5 * dt * k2, fmid)
                k4 = self.rhs(y + dt * k3, f1)
                psi[k + 1] = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown integrator: {self.integrator}")
        return psi

    def output_field(self, F_t: ComplexArray | complex, psi_t: ComplexArray) -> ComplexArray:
        return F_t - np.sqrt(self.cfg.kappa_out_inv_ps) * psi_t


class BalancedDetector:
    def __init__(self, cfg: DetectionConfig, seed: int = 2027) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def vacuum_field(self, n: int) -> ComplexArray:
        if not self.cfg.simulate_vacuum_port:
            return np.zeros(n, dtype=np.complex128)
        v_re = self.rng.normal(scale=self.cfg.sigma_vac, size=n)
        v_im = self.rng.normal(scale=self.cfg.sigma_vac, size=n)
        return ((v_re + 1j * v_im) / np.sqrt(2.0)).astype(np.complex128)

    def currents(self, field_t: ComplexArray, suffix: str = "") -> Dict[str, Array | ComplexArray]:
        v_t = self.vacuum_field(field_t.size)
        b1 = (field_t + v_t) / np.sqrt(2.0)
        b2 = (field_t - v_t) / np.sqrt(2.0)

        scale = self.cfg.responsivity * self.cfg.detection_efficiency
        i1 = scale * np.abs(b1) ** 2
        i2 = scale * np.abs(b2) ** 2
        i_plus = i1 + i2
        i_minus = i1 - i2

        return {
            f"vacuum_t{suffix}": v_t,
            f"b1_t{suffix}": b1,
            f"b2_t{suffix}": b2,
            f"i1_meas_t{suffix}": i1.astype(np.float64),
            f"i2_meas_t{suffix}": i2.astype(np.float64),
            f"i_plus_meas_t{suffix}": i_plus.astype(np.float64),
            f"i_minus_meas_t{suffix}": i_minus.astype(np.float64),
        }

    def reference_currents_for_drive(self, F_t: ComplexArray) -> Dict[str, Array | ComplexArray]:
        raw = self.currents(F_t, suffix="_ref")
        # Preserve original key names used in your old script.
        return {
            "b1_ref_t": raw["b1_t_ref"],
            "b2_ref_t": raw["b2_t_ref"],
            "i_1_meas_ref_t": raw["i1_meas_t_ref"],
            "i_2_meas_ref_t": raw["i2_meas_t_ref"],
            "i_plus_meas_ref_t": raw["i_plus_meas_t_ref"],
            "i_minus_meas_ref_t": raw["i_minus_meas_t_ref"],
        }

    def select_primary_channels(
        self,
        det: Dict[str, np.ndarray],
        det_wn: Dict[str, np.ndarray],
        ref: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if self.cfg.mode == "balanced_sum":
            return {
                "i_meas_t": det["i_plus_meas_t"],
                "i_ref_t": ref["i_plus_meas_ref_t"],
                "i_meas_without_noise_t": det_wn["i_plus_meas_t_wn"],
            }
        if self.cfg.mode == "balanced_diff":
            return {
                "i_meas_t": det["i_minus_meas_t"],
                "i_ref_t": ref["i_minus_meas_ref_t"],
                "i_meas_without_noise_t": det_wn["i_minus_meas_t_wn"],
            }
        raise ValueError("Class refactor currently keeps the same behavior as your script: use balanced_sum or balanced_diff.")


class SpectrumAnalyzer:
    def __init__(self, cfg: SpectrumConfig) -> None:
        self.cfg = cfg

    def psd(self, x: Array, fs_mhz: float) -> Tuple[Array, Array]:
        if SCIPY_AVAILABLE:
            f, pxx = welch(
                x,
                fs=fs_mhz,
                window=self.cfg.window,
                nperseg=min(self.cfg.nperseg, x.size),
                detrend=self.cfg.detrend,
                return_onesided=True,
                scaling="density",
                average=self.cfg.average,
            )
            return f.astype(np.float64), pxx.astype(np.float64)

        n = x.size
        if n < 2:
            raise ValueError("Need at least 2 points to compute a PSD.")
        window = np.hanning(n)
        xw = (x - np.mean(x)) * window
        norm = fs_mhz * np.sum(window**2)
        Xf = np.fft.rfft(xw)
        pxx = (np.abs(Xf) ** 2) / norm
        f = np.fft.rfftfreq(n, d=1.0 / fs_mhz)
        return f.astype(np.float64), pxx.astype(np.float64)

    @staticmethod
    def rbw_average(freqs_mhz: Array, psd: Array, rbw_mhz: float) -> Tuple[Array, Array]:
        if rbw_mhz <= 0 or freqs_mhz.size < 2:
            return freqs_mhz, psd
        df = freqs_mhz[1] - freqs_mhz[0]
        bins = max(1, int(round(rbw_mhz / df)))
        if bins == 1:
            return freqs_mhz, psd
        n = (psd.size // bins) * bins
        return freqs_mhz[:n].reshape(-1, bins).mean(axis=1), psd[:n].reshape(-1, bins).mean(axis=1)


# -----------------------------------------------------------------------------
# High-level experiment
# -----------------------------------------------------------------------------

class BistabilityExperiment:
    def __init__(self, cfg: FullConfig, pump: PumpProtocol | None = None) -> None:
        self.cfg = cfg
        self.pump = pump or PumpProtocol()
        self.cavity = KerrCavity(cfg.cavity, integrator=cfg.sim.integrator)
        self.spectrum = SpectrumAnalyzer(cfg.spectrum)

    def square_cycle_drive(self, t: Array) -> ComplexArray:
        F_t = np.full(t.shape, self.pump.F_low, dtype=np.complex128)
        F_t[t >= self.pump.rise_fraction * self.cfg.sim.duration_ps] = self.pump.F_high
        F_t[t >= self.pump.fall_fraction * self.cfg.sim.duration_ps] = self.pump.F_work
        return F_t

    def prepare_upper_branch(self) -> complex:
        t_prep = time_axis(self.cfg.sim.duration_ps, self.cfg.sim.dt_ps)
        F_prep_t = self.square_cycle_drive(t_prep)
        psi_prep_t = self.cavity.integrate(F_prep_t=F_prep_t, t=t_prep, psi0=0.0 + 0.0j) if False else self.cavity.integrate(t_prep, F_prep_t, psi0=0.0 + 0.0j)
        psi_upper = psi_prep_t[-1]
        print("Prepared upper-branch density:", np.abs(psi_upper) ** 2)
        return complex(psi_upper)

    def compute_bistability_curve(self, F_values: Array, settle_time_ps: float = 2e4) -> Dict[str, Array]:
        t = time_axis(settle_time_ps, self.cfg.sim.dt_ps)
        density_up: list[float] = []
        density_down: list[float] = []

        psi0 = 0.0 + 0.0j
        for F in F_values:
            F_t = np.full(t.shape, F + 0j, dtype=np.complex128)
            psi_t = self.cavity.integrate(t, F_t, psi0=psi0)
            psi0 = complex(psi_t[-1])
            density_up.append(float(np.abs(psi0) ** 2))

        for F in F_values[::-1]:
            F_t = np.full(t.shape, F + 0j, dtype=np.complex128)
            psi_t = self.cavity.integrate(t, F_t, psi0=psi0)
            psi0 = complex(psi_t[-1])
            density_down.append(float(np.abs(psi0) ** 2))

        return {
            "F_up": F_values,
            "density_up": np.array(density_up, dtype=np.float64),
            "F_down": F_values[::-1],
            "density_down": np.array(density_down, dtype=np.float64),
        }

    def _discard_and_downsample(self, results: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        n_full = len(results["t_ps"])
        n0 = int(self.cfg.sim.discard_fraction * n_full)
        step = max(1, int(self.cfg.sim.store_every))

        out: Dict[str, np.ndarray] = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray) and value.shape == results["t_ps"].shape:
                out[key] = value[n0::step]
            else:
                out[key] = value

        fs_mhz = (1.0 / self.cfg.sim.dt_ps) * MHZ_PER_INV_PS
        return out, fs_mhz / step

    def run_upper_branch(self) -> Dict[str, np.ndarray]:
        if self.cfg.detection.mode == "homodyne":
            raise NotImplementedError("Homodyne was not implemented in the original script either. Use balanced_sum or balanced_diff.")

        t_full = time_axis(self.cfg.sim.duration_ps, self.cfg.sim.dt_ps)
        psi_upper = self.prepare_upper_branch()

        self.cfg.cavity.psi0 = psi_upper
        self.cfg.cavity.F_s = self.pump.F_work

        noise = NoiseGenerator(self.cfg.noise)
        F_n, noise_aux = noise.drive_noise(t_full)
        F_t = self.cfg.cavity.F_s + F_n

        psi_t = self.cavity.integrate(t_full, F_t, psi0=psi_upper)
        psi_t_wn = self.cavity.integrate(
            t_full,
            np.full(t_full.size, self.cfg.cavity.F_s, dtype=np.complex128),
            psi0=psi_upper,
        )

        s_out_t = self.cavity.output_field(F_t, psi_t)
        s_out_without_noise_t = self.cavity.output_field(self.cfg.cavity.F_s, psi_t_wn)

        detector = BalancedDetector(self.cfg.detection, seed=self.cfg.noise.seed + 999)
        det = detector.currents(s_out_t, suffix="")
        det_wn = detector.currents(s_out_without_noise_t, suffix="_wn")
        ref = detector.reference_currents_for_drive(F_t)

        results: Dict[str, np.ndarray] = {
            "t_ps": t_full,
            "F_t": F_t,
            "psi_t": psi_t,
            "s_out_t": s_out_t,
            "s_out_without_noise_t": s_out_without_noise_t,
            "amp_noise": noise_aux["amp_noise"],
            "phase_noise": noise_aux["phase_noise"],
        }
        results.update(det)
        results.update(det_wn)
        results.update(ref)
        results.update(detector.select_primary_channels(det, det_wn, ref))

        results, fs_store_mhz = self._discard_and_downsample(results)

        f_meas, psd_meas = self.spectrum.psd(results["i_meas_t"], fs_store_mhz)
        f_drive, psd_drive = self.spectrum.psd(results["i_ref_t"], fs_store_mhz)
        f_wn, psd_wn = self.spectrum.psd(results["i_meas_without_noise_t"], fs_store_mhz)

        x_in, p_in = complex_to_quadratures(results["F_t"])
        x_cav, p_cav = complex_to_quadratures(results["psi_t"])
        x_out, p_out = complex_to_quadratures(results["s_out_t"])

        results.update({
            "x_in": x_in,
            "p_in": p_in,
            "x_cav": x_cav,
            "p_cav": p_cav,
            "x_out": x_out,
            "p_out": p_out,
            "freqs_meas_mhz": f_meas,
            "psd_meas": psd_meas,
            "freqs_drive_mhz": f_drive,
            "psd_drive": psd_drive,
            "freqs_wn_mhz": f_wn,
            "psd_wn": psd_wn,
            "fs_store_mhz": np.array([fs_store_mhz], dtype=np.float64),
            "F_low": np.array([self.pump.F_low], dtype=np.complex128),
            "F_high": np.array([self.pump.F_high], dtype=np.complex128),
            "F_work": np.array([self.pump.F_work], dtype=np.complex128),
        })
        return results

    def run_full(self, include_bistability: bool = True, F_values: Optional[Array] = None) -> Dict[str, np.ndarray]:
        results = self.run_upper_branch()
        if include_bistability:
            if F_values is None:
                F_values = np.linspace(0.0, 3.0, 100)
            bistab = self.compute_bistability_curve(F_values)
            results.update({
                "bistab_F_up": bistab["F_up"],
                "bistab_density_up": bistab["density_up"],
                "bistab_F_down": bistab["F_down"],
                "bistab_density_down": bistab["density_down"],
            })
        return results

    def save(self, path: str | Path, results: Dict[str, np.ndarray]) -> None:
        save_results_npz(path, self.cfg, results)


# -----------------------------------------------------------------------------
# Compatibility wrappers: old functional API still works
# -----------------------------------------------------------------------------

def generate_band_limited_real_gaussian_noise(n: int, dt_ps: float, cutoff_mhz: float, rms: float, rng: np.random.Generator) -> Array:
    return NoiseGenerator.band_limited_real_gaussian(n, dt_ps, cutoff_mhz, rms, rng)


def generate_drive_noise(t: Array, cfg: NoiseConfig) -> Tuple[ComplexArray, Dict[str, Array]]:
    return NoiseGenerator(cfg).drive_noise(t)


def cavity_rhs(
    psi: complex,
    F: complex,
    detuning_inv_ps: float,
    nonlinearity_inv_ps: float,
    kappa_out_inv_ps: float,
    loss_inv_ps: float,
    cfg: CavityConfig,
) -> complex:
    return (
        1j * detuning_inv_ps * psi
        - 1j * nonlinearity_inv_ps * (abs(psi) ** 2) * psi
        - 0.5 * loss_inv_ps * psi
        + np.sqrt(kappa_out_inv_ps) * F
    )


def integrate_cavity(t: Array, F_t: ComplexArray, cfg: CavityConfig, integrator: str = "rk4") -> ComplexArray:
    return KerrCavity(cfg, integrator=integrator).integrate(t, F_t)


def output_field(F_t: ComplexArray | complex, psi_t: ComplexArray, cfg: CavityConfig) -> ComplexArray:
    return KerrCavity(cfg).output_field(F_t, psi_t)


def generate_vacuum_field(n: int, sigma_vac: float, rng: np.random.Generator) -> ComplexArray:
    v_re = rng.normal(scale=sigma_vac, size=n)
    v_im = rng.normal(scale=sigma_vac, size=n)
    return ((v_re + 1j * v_im) / np.sqrt(2.0)).astype(np.complex128)


def balanced_direct_detection_currents(s_out_t: ComplexArray, cfg: DetectionConfig, rng: Optional[np.random.Generator] = None, dt_ps: Optional[float] = None) -> Dict[str, np.ndarray]:
    det = BalancedDetector(cfg, seed=2027)
    if rng is not None:
        det.rng = rng
    return det.currents(s_out_t, suffix="")


def balanced_direct_detection_currents_without_noise(s_out_without_noise_t: ComplexArray, cfg: DetectionConfig, rng: Optional[np.random.Generator] = None, dt_ps: Optional[float] = None) -> Dict[str, np.ndarray]:
    det = BalancedDetector(cfg, seed=2027)
    if rng is not None:
        det.rng = rng
    return det.currents(s_out_without_noise_t, suffix="_wn")


def balanced_current_for_drive_noise(F_t: ComplexArray, cfg: DetectionConfig, rng: Optional[np.random.Generator] = None, dt_ps: Optional[float] = None) -> Dict[str, np.ndarray]:
    det = BalancedDetector(cfg, seed=2027)
    if rng is not None:
        det.rng = rng
    return det.reference_currents_for_drive(F_t)


def compute_psd(x: Array, fs_mhz: float, cfg: SpectrumConfig) -> Tuple[Array, Array]:
    return SpectrumAnalyzer(cfg).psd(x, fs_mhz)


def rbw_average_psd(freqs_mhz: Array, psd: Array, rbw_mhz: float) -> Tuple[Array, Array]:
    return SpectrumAnalyzer.rbw_average(freqs_mhz, psd, rbw_mhz)


def make_square_cycle_drive(t: Array, F_low: complex, F_high: complex, F_work: complex, t_rise_ps: float, t_fall_ps: float) -> ComplexArray:
    F_t = np.full(t.shape, F_low, dtype=np.complex128)
    F_t[t >= t_rise_ps] = F_high
    F_t[t >= t_fall_ps] = F_work
    return F_t


def prepare_upper_branch(cfg: FullConfig, F_low: complex, F_high: complex, F_work: complex) -> complex:
    pump = PumpProtocol(F_low=F_low, F_high=F_high, F_work=F_work)
    return BistabilityExperiment(cfg, pump).prepare_upper_branch()


def compute_bistability_curve(cfg: FullConfig, F_values: Array, settle_time_ps: float = 2e4) -> Dict[str, Array]:
    return BistabilityExperiment(cfg).compute_bistability_curve(F_values, settle_time_ps=settle_time_ps)


def run_simulation_with_upper_branch(cfg: FullConfig, F_low: complex, F_high: complex, F_work: complex) -> Dict[str, np.ndarray]:
    pump = PumpProtocol(F_low=F_low, F_high=F_high, F_work=F_work)
    return BistabilityExperiment(cfg, pump).run_upper_branch()


# -----------------------------------------------------------------------------
# Lightweight plotting helpers kept for convenience
# -----------------------------------------------------------------------------

def plot_bistability_from_results(results: Dict[str, np.ndarray]) -> plt.Figure:
    fig = plt.figure(figsize=(7, 5))
    plt.scatter(results["bistab_F_up"], results["bistab_density_up"], label="Sweep up", marker="x")
    plt.scatter(results["bistab_F_down"], results["bistab_density_down"], label="Sweep down", marker="+")
    if "F_work" in results and "psi_t" in results:
        plt.scatter([np.real(results["F_work"][0])], [np.abs(results["psi_t"][0]) ** 2], s=50, label="Initial state at F_work")
    plt.xlabel("Pump amplitude F")
    plt.ylabel(r"Intracavity density $|\psi|^2$")
    plt.title("Polariton bistability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    cfg = FullConfig()
    cfg.detection.mode = "balanced_sum"
    cfg.detection.add_shot_noise = True
    cfg.detection.shot_noise_mode = "photocurrent"
    cfg.detection.shot_noise_gain_per_current = 1.0
    cfg.detection.shot_noise_use_instantaneous_photocurrent = True
    cfg.detection.electronic_noise_psd_per_mhz = 0.0

    pump = PumpProtocol(F_low=0.3 + 0j, F_high=1.2 + 0j, F_work=0.75 + 0j)
    exp = BistabilityExperiment(cfg, pump)

    results = exp.run_full(include_bistability=True, F_values=np.linspace(0.0, 3.0, 100))

    print("Input drive quadrature stats:")
    for k, v in estimate_quadrature_variances(results["F_t"]).items():
        print(f"  {k:>10s} = {v:.6g}")
    print("\nOutput field quadrature stats:")
    for k, v in estimate_quadrature_variances(results["s_out_t"]).items():
        print(f"  {k:>10s} = {v:.6g}")

    out_path = Path("Results/polariton_homodyne_results_balanced_both_7.npz")
    exp.save(out_path, results)
    print(f"\nSaved results to {out_path}")

    fig = plot_bistability_from_results(results)
    plt.show()


if __name__ == "__main__":
    main()
