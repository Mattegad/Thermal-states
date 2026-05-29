from __future__ import annotations

"""
polariton_bistability_core.py

Core object-oriented structure for the polariton microcavity bistability
simulation and replot pipeline.

This file contains:
- configuration dataclasses
- physics components: NoiseGenerator, KerrCavity, BalancedDetector
- SpectrumAnalyzer
- SimulationResults with load/save/reconstruct helpers
- BistabilityExperiment
- ResultPlotter

The saved .npz keys intentionally remain compatible with the previous
Replot_bistability.py convention:
F_t, psi_t, s_out_t, i_plus_meas_t, i_minus_meas_t, psd_meas, etc.
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


# =============================================================================
# Config containers
# =============================================================================

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
    strength_amp: float = 0.02 * 10 ** (5.0 / 20.0)
    strength_phase: float = 0.02 * 10 ** (5.0 / 20.0)
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


# =============================================================================
# Small utilities
# =============================================================================

def clone_config(base_cfg: FullConfig) -> FullConfig:
    return copy.deepcopy(base_cfg)


def time_axis(duration_ps: float, dt_ps: float) -> Array:
    n = int(np.round(duration_ps / dt_ps))
    return np.arange(n, dtype=np.float64) * dt_ps


def complex_to_quadratures(z: np.ndarray) -> Tuple[Array, Array]:
    return np.sqrt(2.0) * np.real(z), np.sqrt(2.0) * np.imag(z)


def quadrature_projection(z: np.ndarray, theta: float) -> Array:
    return np.sqrt(2.0) * np.real(np.exp(-1j * theta) * z)


def estimate_quadrature_variances(z: np.ndarray) -> Dict[str, float]:
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


def restore_complex(obj: Any) -> Any:
    if isinstance(obj, dict):
        if obj.get("__complex__"):
            return complex(obj["real"], obj["imag"])
        return {k: restore_complex(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [restore_complex(v) for v in obj]
    return obj


def config_metadata(cfg: FullConfig) -> Dict[str, Any]:
    return {
        "noise": asdict(cfg.noise),
        "cavity": asdict(cfg.cavity),
        "sim": asdict(cfg.sim),
        "detection": asdict(cfg.detection),
        "spectrum": asdict(cfg.spectrum),
    }


# =============================================================================
# Spectrum analyzer / generic analysis
# =============================================================================

class SpectrumAnalyzer:
    def __init__(self, cfg: SpectrumConfig | None = None) -> None:
        self.cfg = cfg or SpectrumConfig()

    def psd(self, x: np.ndarray, fs_mhz: float) -> Tuple[Array, Array]:
        x = np.asarray(x, dtype=np.float64)

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
    def rbw_average(freqs_mhz: np.ndarray, psd: np.ndarray, rbw_mhz: float) -> Tuple[Array, Array]:
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

    @staticmethod
    def integrated_band_power(
        freqs_mhz: np.ndarray,
        psd: np.ndarray,
        fmin_mhz: Optional[float] = None,
        fmax_mhz: Optional[float] = None,
    ) -> float:
        mask = np.ones_like(freqs_mhz, dtype=bool)
        if fmin_mhz is not None:
            mask &= freqs_mhz >= fmin_mhz
        if fmax_mhz is not None:
            mask &= freqs_mhz <= fmax_mhz
        if not np.any(mask):
            return float("nan")
        return float(np.trapz(psd[mask], freqs_mhz[mask]))


# =============================================================================
# Results object
# =============================================================================

@dataclass
class SimulationResults:
    """Container around a flat result dictionary.

    The simulation and replot pipeline share this object, so the replot script
    does not redefine loading, quadratures, PSDs, sample-rate inference, etc.
    """

    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[Path] = None

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def get(self, key: str, default: Any = None) -> Any:
        return self.arrays.get(key, default)

    def add(self, **arrays: np.ndarray) -> None:
        self.arrays.update(arrays)

    def to_dict(self) -> Dict[str, np.ndarray]:
        return dict(self.arrays)

    @classmethod
    def from_core(
        cls,
        *,
        t_ps: Array,
        F_t: ComplexArray,
        psi_t: ComplexArray,
        s_out_t: ComplexArray,
        s_out_without_noise_t: ComplexArray,
        amp_noise: Array,
        phase_noise: Array,
        metadata: Optional[Dict[str, Any]] = None,
        **extra: np.ndarray,
    ) -> "SimulationResults":
        arrays: Dict[str, np.ndarray] = {
            "t_ps": t_ps,
            "F_t": F_t,
            "psi_t": psi_t,
            "s_out_t": s_out_t,
            "s_out_without_noise_t": s_out_without_noise_t,
            "amp_noise": amp_noise,
            "phase_noise": phase_noise,
        }
        arrays.update(extra)
        return cls(arrays=arrays, metadata=metadata or {})

    @classmethod
    def load_npz(cls, path: str | Path, build_missing: bool = True) -> "SimulationResults":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find result file: {path}")

        data = np.load(path, allow_pickle=True)
        metadata: Dict[str, Any] = {}
        arrays: Dict[str, np.ndarray] = {}

        for key in data.files:
            if key == "metadata_json":
                raw = data[key]
                if isinstance(raw, np.ndarray):
                    raw = raw.item()
                metadata = restore_complex(json.loads(str(raw)))
            else:
                arrays[key] = data[key]

        out = cls(arrays=arrays, metadata=metadata, source_path=path)
        if build_missing:
            out.build_missing_channels()
        return out

    def save_npz(self, path: str | Path, cfg: Optional[FullConfig] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        metadata = config_metadata(cfg) if cfg is not None else self.metadata
        np.savez_compressed(
            path,
            metadata_json=json.dumps(_json_safe(metadata), indent=2),
            **self.arrays,
        )

    @property
    def detection_mode(self) -> str:
        det = self.metadata.get("detection", {})
        if isinstance(det, dict):
            return str(det.get("mode", "unknown"))
        return "unknown"

    def infer_sample_rate_mhz(self) -> float:
        if "fs_store_mhz" in self.arrays:
            val = self.arrays["fs_store_mhz"]
            if np.ndim(val) == 0:
                return float(val)
            return float(np.ravel(val)[0])
        if "t_ps" in self.arrays and len(self.arrays["t_ps"]) > 1:
            dt_ps = float(self.arrays["t_ps"][1] - self.arrays["t_ps"][0])
            return (1.0 / dt_ps) * MHZ_PER_INV_PS
        raise ValueError("Could not infer sample rate from results.")

    def build_missing_channels(self, spectrum: Optional[SpectrumAnalyzer] = None) -> None:
        """Reconstruct cheap derived channels only if they are missing."""
        spectrum = spectrum or SpectrumAnalyzer()

        if "s_out_t" in self.arrays:
            if "x_out" not in self.arrays or "p_out" not in self.arrays:
                self.arrays["x_out"], self.arrays["p_out"] = complex_to_quadratures(self.arrays["s_out_t"])

        if "F_t" in self.arrays:
            if "x_in" not in self.arrays or "p_in" not in self.arrays:
                self.arrays["x_in"], self.arrays["p_in"] = complex_to_quadratures(self.arrays["F_t"])

        if "psi_t" in self.arrays:
            if "x_cav" not in self.arrays or "p_cav" not in self.arrays:
                self.arrays["x_cav"], self.arrays["p_cav"] = complex_to_quadratures(self.arrays["psi_t"])

        if "i_meas_t" not in self.arrays:
            if self.detection_mode == "balanced_sum" and "i_plus_meas_t" in self.arrays:
                self.arrays["i_meas_t"] = self.arrays["i_plus_meas_t"]
            elif self.detection_mode == "balanced_diff" and "i_minus_meas_t" in self.arrays:
                self.arrays["i_meas_t"] = self.arrays["i_minus_meas_t"]

        if "freqs_meas_mhz" not in self.arrays and "i_meas_t" in self.arrays:
            f, p = spectrum.psd(self.arrays["i_meas_t"], self.infer_sample_rate_mhz())
            self.arrays["freqs_meas_mhz"] = f
            self.arrays["psd_meas"] = p

    def get_psd(
        self,
        signal_key: str,
        freq_key: Optional[str] = None,
        psd_key: Optional[str] = None,
        spectrum: Optional[SpectrumAnalyzer] = None,
        force_recompute: bool = False,
    ) -> Tuple[Array, Array]:
        """Reuse saved PSD if available; otherwise compute from a saved time trace."""
        if (
            not force_recompute
            and freq_key is not None
            and psd_key is not None
            and freq_key in self.arrays
            and psd_key in self.arrays
        ):
            return self.arrays[freq_key], self.arrays[psd_key]

        if signal_key not in self.arrays:
            raise KeyError(f"Missing signal '{signal_key}' for PSD computation.")

        spectrum = spectrum or SpectrumAnalyzer()
        x = np.asarray(self.arrays[signal_key], dtype=np.float64)
        return spectrum.psd(x - np.mean(x), self.infer_sample_rate_mhz())

    def print_summary(self) -> None:
        print("\n=== Metadata summary ===")
        if not self.metadata:
            print("No metadata found.")
        else:
            for section, content in self.metadata.items():
                print(f"[{section}]")
                if isinstance(content, dict):
                    for k, v in content.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {content}")

        print("\n=== Saved arrays ===")
        for k, v in self.arrays.items():
            print(f"{k:>24s}: shape={getattr(v, 'shape', None)}, dtype={getattr(v, 'dtype', None)}")

        for key in ("F_t", "psi_t", "s_out_t"):
            if key in self.arrays:
                print(f"\n=== Quadrature stats for {key} ===")
                for name, value in estimate_quadrature_variances(self.arrays[key]).items():
                    print(f"  {name:>10s} = {value:.6g}")


# =============================================================================
# Physics components
# =============================================================================

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

    def currents(self, field_t: ComplexArray, suffix: str = "") -> Dict[str, np.ndarray]:
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

    def reference_currents_for_drive(self, F_t: ComplexArray) -> Dict[str, np.ndarray]:
        raw = self.currents(F_t, suffix="_ref")
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

        raise NotImplementedError("Use balanced_sum or balanced_diff. Homodyne was not implemented in your previous script.")


# =============================================================================
# High-level experiment
# =============================================================================

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
        psi_prep_t = self.cavity.integrate(t_prep, F_prep_t, psi0=0.0 + 0.0j)
        psi_upper = complex(psi_prep_t[-1])

        print("Prepared upper-branch density:", np.abs(psi_upper) ** 2)
        return psi_upper

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

    def _discard_and_downsample(self, arrays: Dict[str, np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        n_full = len(arrays["t_ps"])
        n0 = int(self.cfg.sim.discard_fraction * n_full)
        step = max(1, int(self.cfg.sim.store_every))

        out: Dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            if isinstance(value, np.ndarray) and value.shape == arrays["t_ps"].shape:
                out[key] = value[n0::step]
            else:
                out[key] = value

        fs_mhz = (1.0 / self.cfg.sim.dt_ps) * MHZ_PER_INV_PS
        return out, fs_mhz / step

    def run_upper_branch(self) -> SimulationResults:
        if self.cfg.detection.mode == "homodyne":
            raise NotImplementedError("Homodyne is not implemented here yet. Use balanced_sum or balanced_diff.")

        t_full = time_axis(self.cfg.sim.duration_ps, self.cfg.sim.dt_ps)
        psi_upper = self.prepare_upper_branch()

        self.cfg.cavity.psi0 = psi_upper
        self.cfg.cavity.F_s = self.pump.F_work

        noise = NoiseGenerator(self.cfg.noise)
        F_n, noise_aux = noise.drive_noise(t_full)
        F_t = self.cfg.cavity.F_s + F_n

        psi_t = self.cavity.integrate(t_full, F_t, psi0=psi_upper)

        F_wn = np.full(t_full.size, self.cfg.cavity.F_s, dtype=np.complex128)
        psi_t_wn = self.cavity.integrate(t_full, F_wn, psi0=psi_upper)

        s_out_t = self.cavity.output_field(F_t, psi_t)
        s_out_without_noise_t = self.cavity.output_field(self.cfg.cavity.F_s, psi_t_wn)

        detector = BalancedDetector(self.cfg.detection, seed=self.cfg.noise.seed + 999)

        det = detector.currents(s_out_t, suffix="")
        det_wn = detector.currents(s_out_without_noise_t, suffix="_wn")
        ref = detector.reference_currents_for_drive(F_t)

        raw: Dict[str, np.ndarray] = {
            "t_ps": t_full,
            "F_t": F_t,
            "psi_t": psi_t,
            "s_out_t": s_out_t,
            "s_out_without_noise_t": s_out_without_noise_t,
            "amp_noise": noise_aux["amp_noise"],
            "phase_noise": noise_aux["phase_noise"],
        }
        raw.update(det)
        raw.update(det_wn)
        raw.update(ref)
        raw.update(detector.select_primary_channels(det, det_wn, ref))

        arrays, fs_store_mhz = self._discard_and_downsample(raw)

        f_meas, psd_meas = self.spectrum.psd(arrays["i_meas_t"], fs_store_mhz)
        f_drive, psd_drive = self.spectrum.psd(arrays["i_ref_t"], fs_store_mhz)
        f_wn, psd_wn = self.spectrum.psd(arrays["i_meas_without_noise_t"], fs_store_mhz)

        x_in, p_in = complex_to_quadratures(arrays["F_t"])
        x_cav, p_cav = complex_to_quadratures(arrays["psi_t"])
        x_out, p_out = complex_to_quadratures(arrays["s_out_t"])

        arrays.update({
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

        return SimulationResults(arrays=arrays, metadata=config_metadata(self.cfg))

    def run_full(self, include_bistability: bool = True, F_values: Optional[Array] = None) -> SimulationResults:
        results = self.run_upper_branch()

        if include_bistability:
            if F_values is None:
                F_values = np.linspace(0.0, 3.0, 100)

            bistab = self.compute_bistability_curve(F_values)
            results.add(
                bistab_F_up=bistab["F_up"],
                bistab_density_up=bistab["density_up"],
                bistab_F_down=bistab["F_down"],
                bistab_density_down=bistab["density_down"],
            )

        return results


# =============================================================================
# Plotting
# =============================================================================

class ResultPlotter:
    def __init__(self, results: SimulationResults, spectrum: Optional[SpectrumAnalyzer] = None) -> None:
        self.results = results
        self.results.build_missing_channels(spectrum=spectrum)
        self.spectrum = spectrum or SpectrumAnalyzer()

    @property
    def a(self) -> Dict[str, np.ndarray]:
        return self.results.arrays

    def time_traces(self, max_points: int = 6000) -> plt.Figure:
        a = self.a
        t = a["t_ps"]
        step = max(1, len(t) // max_points)
        sl = slice(None, None, step)

        mode = self.results.detection_mode
        nrows = 6 if mode != "homodyne" else 5
        fig, axes = plt.subplots(nrows, 1, figsize=(12, 3 * nrows), sharex=True)
        row = 0

        if "amp_noise" in a:
            axes[row].plot(t[sl] * 1e6, a["amp_noise"][sl], label="Amplitude noise")
        if "phase_noise" in a:
            axes[row].plot(t[sl] * 1e6, a["phase_noise"][sl], label="Phase noise")
        axes[row].set_ylabel("Noise")
        axes[row].legend()
        axes[row].grid(True, alpha=0.3)
        row += 1

        axes[row].plot(t[sl] * 1e6, np.real(a["F_t"][sl]), label="Re F(t)")
        axes[row].plot(t[sl] * 1e6, np.imag(a["F_t"][sl]), label="Im F(t)")
        axes[row].set_ylabel("Drive")
        axes[row].legend()
        axes[row].grid(True, alpha=0.3)
        row += 1

        axes[row].plot(t[sl] * 1e6, np.real(a["psi_t"][sl]), label="Re ψ(t)")
        axes[row].plot(t[sl] * 1e6, np.imag(a["psi_t"][sl]), label="Im ψ(t)")
        axes[row].set_ylabel("Intracavity")
        axes[row].legend()
        axes[row].grid(True, alpha=0.3)
        row += 1

        axes[row].plot(t[sl] * 1e6, np.real(a["s_out_t"][sl]), label="Re s_out(t)")
        axes[row].plot(t[sl] * 1e6, np.imag(a["s_out_t"][sl]), label="Im s_out(t)")
        axes[row].set_ylabel("Output")
        axes[row].legend()
        axes[row].grid(True, alpha=0.3)
        row += 1

        if mode == "homodyne":
            if "i_meas_t" in a:
                axes[row].plot(t[sl] * 1e6, a["i_meas_t"][sl], label="i_meas(t)")
            axes[row].set_ylabel("Photocurrent")
            axes[row].legend()
            axes[row].grid(True, alpha=0.3)
        else:
            if "i1_meas_t" in a:
                axes[row].plot(t[sl] * 1e6, a["i1_meas_t"][sl], label="i1")
            if "i2_meas_t" in a:
                axes[row].plot(t[sl] * 1e6, a["i2_meas_t"][sl], label="i2")
            axes[row].set_ylabel("Diodes")
            axes[row].legend()
            axes[row].grid(True, alpha=0.3)
            row += 1

            if "i_plus_meas_t" in a:
                axes[row].plot(t[sl] * 1e6, a["i_plus_meas_t"][sl], label="i+")
            if "i_minus_meas_t" in a:
                axes[row].plot(t[sl] * 1e6, a["i_minus_meas_t"][sl], label="i-")
            axes[row].set_ylabel("Balanced")
            axes[row].legend()
            axes[row].grid(True, alpha=0.3)

        axes[-1].set_xlabel(r"Time ($\mu$s)")
        fig.tight_layout()
        return fig

    def quadratures_vs_time(self, max_points: int = 6000) -> plt.Figure:
        a = self.a
        t = a["t_ps"]
        step = max(1, len(t) // max_points)
        sl = slice(None, None, step)

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(t[sl] * 1e6, a["x_in"][sl], label="X_in")
        axes[0].plot(t[sl] * 1e6, a["p_in"][sl], label="P_in")
        axes[0].set_ylabel("Input")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t[sl] * 1e6, a["x_cav"][sl], label="X_cav")
        axes[1].plot(t[sl] * 1e6, a["p_cav"][sl], label="P_cav")
        axes[1].set_ylabel("Cavity")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t[sl] * 1e6, a["x_out"][sl], label="X_out")
        axes[2].plot(t[sl] * 1e6, a["p_out"][sl], label="P_out")
        axes[2].set_ylabel("Output")
        axes[2].set_xlabel(r"Time ($\mu$s)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def phase_space(self, max_points: int = 50000) -> plt.Figure:
        a = self.a
        channels = [
            ("x_in", "p_in", "Input drive quadratures", "X_in", "P_in"),
            ("x_cav", "p_cav", "Intracavity quadratures", "X_cav", "P_cav"),
            ("x_out", "p_out", "Output field quadratures", "X_out", "P_out"),
        ]
        available = [c for c in channels if c[0] in a and c[1] in a]

        fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for ax, (xk, pk, title, xlabel, ylabel) in zip(axes, available):
            n = len(a[xk])
            step = max(1, n // max_points)
            sl = slice(None, None, step)
            ax.scatter(a[xk][sl], a[pk][sl], s=2, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig

    def spectra(
        self,
        rbw_mhz: float = 0.0,
        fmin_mhz: Optional[float] = None,
        fmax_mhz: Optional[float] = 2000.0,
        loglog: bool = False,
    ) -> plt.Figure:
        a = self.a
        f, p = self.results.get_psd(
            signal_key="i_meas_t",
            freq_key="freqs_meas_mhz",
            psd_key="psd_meas",
            spectrum=self.spectrum,
        )

        if rbw_mhz > 0:
            f, p = self.spectrum.rbw_average(f, p, rbw_mhz)

        mask = np.ones_like(f, dtype=bool)
        if fmin_mhz is not None:
            mask &= f >= fmin_mhz
        if fmax_mhz is not None:
            mask &= f <= fmax_mhz

        fig = plt.figure(figsize=(10, 6))
        if loglog:
            plt.loglog(f[mask], p[mask], label="Measured PSD")
        else:
            plt.semilogy(f[mask], p[mask], label="Measured PSD")
        plt.xlabel("Analysis frequency (MHz)")
        plt.ylabel("PSD")
        plt.title("Spectrum analyzer trace")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        fig.tight_layout()
        return fig

    def balanced_psds(
        self,
        rbw_mhz: float = 4.0,
        fmin_mhz: Optional[float] = 1e-3,
        fmax_mhz: Optional[float] = 2000.0,
        shot_noise_channel: str = "i_minus_meas_t_wn",
        eps: float = 1e-30,
    ) -> Optional[plt.Figure]:
        a = self.a
        channels = [
            "i_plus_meas_t",
            "i_minus_meas_t",
            "i_plus_meas_t_wn",
            "i_minus_meas_t_wn",
            "i_plus_meas_ref_t",
            "i_minus_meas_ref_t",
        ]
        available = [k for k in channels if k in a]

        if not available:
            return None

        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for key in available:
            f, p = self.results.get_psd(key, spectrum=self.spectrum, force_recompute=True)
            if rbw_mhz > 0:
                f, p = self.spectrum.rbw_average(f, p, rbw_mhz)
            curves[key] = (f, p)

        if shot_noise_channel in curves:
            _, p_ref = curves[shot_noise_channel]
            p_ref = np.maximum(p_ref, eps)
        else:
            p_ref = None

        labels = {
            "i_plus_meas_t": "i+ meas",
            "i_minus_meas_t": "i- meas",
            "i_plus_meas_t_wn": "i+ without noise",
            "i_minus_meas_t_wn": "i- without noise",
            "i_plus_meas_ref_t": "i+ input ref",
            "i_minus_meas_ref_t": "i- input ref",
        }

        fig = plt.figure(figsize=(10, 6))

        for key, (f, p) in curves.items():
            mask = np.ones_like(f, dtype=bool)
            if fmin_mhz is not None:
                mask &= f >= fmin_mhz
            if fmax_mhz is not None:
                mask &= f <= fmax_mhz

            if p_ref is not None and len(p_ref) == len(p):
                y = 10.0 * np.log10(np.maximum(p, eps) / p_ref)
                ylabel = f"Noise power relative to {shot_noise_channel} (dB)"
            else:
                y = np.maximum(p, eps)
                ylabel = "PSD"

            plt.plot(f[mask], y[mask], label=labels.get(key, key))

        plt.xlabel("Analysis frequency (MHz)")
        plt.ylabel(ylabel)
        plt.title("Balanced detection PSDs")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        fig.tight_layout()
        return fig

    def bistability(self) -> Optional[plt.Figure]:
        a = self.a
        needed = ["bistab_F_up", "bistab_density_up", "bistab_F_down", "bistab_density_down"]
        if not all(k in a for k in needed):
            print("No bistability data found in results.")
            return None

        fig = plt.figure(figsize=(7, 5))
        plt.scatter(np.real(a["bistab_F_up"]), a["bistab_density_up"], label="Sweep up", marker="x")
        plt.scatter(np.real(a["bistab_F_down"]), a["bistab_density_down"], label="Sweep down", marker="+")

        if "F_work" in a and "psi_t" in a:
            plt.scatter(
                [np.real(a["F_work"][0])],
                [np.abs(a["psi_t"][0]) ** 2],
                s=50,
                label="Initial state at F_work",
                zorder=5,
            )

        plt.xlabel("Pump amplitude F")
        plt.ylabel(r"Intracavity density $|\psi|^2$")
        plt.title("Polariton bistability")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def save_figure(fig: plt.Figure, outdir: str | Path, name: str, dpi: int = 180) -> None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outdir / name, dpi=dpi, bbox_inches="tight")
