from __future__ import annotations

"""
Simulation of a single-mode polariton microcavity driven by a coherent field plus
band-limited classical noise prepared with an electro-optic modulator (EOM), and
post-processed through balanced homodyne detection to emulate what a spectrum
analyzer would measure.

Model summary
-------------
1) Optical input field (slow envelope in the rotating frame of the carrier):
       F(t) = F_s + F_n(t)
   where F_n(t) is either amplitude noise (real) or phase noise (imaginary), or
   both. The noise is generated in Fourier space with a flat spectrum up to a
   cutoff frequency.

2) Polariton single-mode mean-field dynamics (Gross-Pitaevskii / driven Kerr mode):
       i dψ/dt = [ -Δ + U |ψ|^2 - i γ/2 ] ψ + F(t)

3) Input-output relation for the detected optical field:
       s_out(t) = F(t) - sqrt(κ_out) ψ(t)
   This is the standard single-sided cavity convention up to a choice of units.
   You may adapt the sign/prefactor to your setup if needed.

4) Balanced homodyne detection with LO phase θ:
       i_hom(t) ∝ 2 |β_LO| Re[ exp(-i θ) s_out(t) ]
   The electronic AC component is then sent to a spectrum analyzer.

5) Spectrum analyzer trace:
       PSD of i_hom(t) computed with Welch's method.
   Optionally, a shot-noise floor is added for visualization/comparison.

This code is intentionally modular and verbose so it can be adapted to many
experimental conventions.

Noise model
--------------------------
This version distinguishes clearly:
-Optical shot noise, whose PSD is made proportional to the relevant detected
photocurrent,
-Additive electronics noise, modeled as an additional stationary white floor


Units convention used here
--------------------------
Time:
    ps

Dynamical coefficients:
    ps^-1

FFT / spectrum display:
    MHz

Meaning:
    1 / ps = 1 THz = 1e6 MHz

Model equation:
    i dψ/dt = [ -Δ + U |ψ|^2 - i γ/2 ] ψ + F(t)

with:
    Δ = δ_meV / ħ              [ps^-1]
    U = g_meV_um2 / ħ          [ps^-1 µm^2]
    γ = Γ_meV / ħ              [ps^-1]
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional, Literal
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# SciPy is convenient for Welch PSD. If unavailable, a fallback FFT-based PSD is provided.
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

# -----------------------------------------------------------------------------
# Physical constants / Unit conversions
# -----------------------------------------------------------------------------

HBAR_MEV_PS = 0.6582119514  # ħ in meV*ps
MHZ_PER_INV_PS = 1.0e6  # 1/(ps) in MHz
INV_PS_PER_MHZ = 1.0 / MHZ_PER_INV_PS


# -----------------------------------------------------------------------------
# Parameter containers
# -----------------------------------------------------------------------------

@dataclass
class CavityConfig:
    """Single-mode driven polariton cavity parameters in the rotating frame.

    Equation:
        i dψ/dt = [ -Δ + U |ψ|^2 - i γ/2 ] ψ + F(t)

    All dynamical coefficients are in units of ps^-1. To convert from meV, use ħ = 0.658 meV*ps.
    """
    detuning_inv_ps: float = 1.4e-1 / HBAR_MEV_PS   # Δ = δ_meV / ħ
    nonlinearity_inv_ps: float =  1.2e-2 / HBAR_MEV_PS  # U = g_meV_um2 / ħ
    loss_inv_ps: float = 7e-2 / HBAR_MEV_PS       # γ
    kappa_out_inv_ps: float = 7e-2 / HBAR_MEV_PS  # output coupling used in input-output relation
    F_s: complex = 0.7 + 0j      # coherent drive amplitude
    psi0: complex = 0.0 + 0.0j   # initial intracavity field


@dataclass
class SimulationConfig:
    """Global simulation parameters."""
    duration_ps: float = 1.0e6        # total simulated time in ps -> resolution in MHz is ~ 1/duration_ps
    dt_ps: float = 1.0              # timestep; sampling rate = 1/dt
    discard_fraction: float = 0.1     # discard initial transient before PSD
    integrator: Literal["rk4", "heun", "euler"] = "rk4"
    store_every: int = 250              # downsampling factor for storage/PSD


@dataclass
class DetectionConfig:
    """Detection configuration.
    mode :
        -'homodyne' -> standard balanced homodyne detection with LO phase control
        -'balanced_sum' -> sum of the two photodiode currents (LO phase irrelevant)
        -'balanced_diff' -> difference of the two photodiode currents (LO phase irrelevant)
    """
    mode: DetectionMode = "balanced_sum"

    # Homodyne parameters
    lo_amplitude: float = 50.0        # |β_LO| (arbitrary units)
    lo_phase_rad: float = np.pi / 2   # θ = π/2 detects phase quadrature

    # General detection parameters
    detection_efficiency: float = 1.0
    responsivity: float = 1.0         # arbitrary scale factor for current

    # Noise model
    add_shot_noise: bool = True
    shot_noise_mode: ShotNoiseMode = "photocurrent"

    # Legacy phenomenological stationary white floor used if shot_noise_mode == "fixed"
    shot_noise_psd_per_mhz: float = 0.0

    # Effective optical shot-noise coefficient in simulation units:
    # PSG(current)/MHz = shot_noise_gain_per_current * photocurrent
    shot_noise_gain_per_current: float = 1.0

    # If True: local time-dependent shot-noise variance from instantaneous photocurrent
    # If False: use fixed variance from mean photocurrent
    shot_noise_use_instantaneous_photocurrent: bool = False

    # Additional electronics white flor added on top
    electronic_noise_psd_per_mhz: float = 0.0

    # Vacuum port for balanced detection
    simulate_vacuum_port: bool = True
    sigma_vac: float = 0.02


@dataclass
class NoiseConfig:
    """Configuration of the band-limited Gaussian drive noise.

    Attributes
    ----------
    mode:
        'amplitude' -> F_n(t) is real
        'phase'     -> F_n(t) is purely imaginary
        'both'      -> independent noise on amplitude and phase quadratures
    cutoff_hz:
        Noise bandwidth. The generated PSD is approximately flat for |f| <= cutoff_hz.
    strength_amp:
        RMS scale of the amplitude-noise contribution to F_n(t).
    strength_phase:
        RMS scale of the phase-noise contribution to F_n(t).
    seed:
        RNG seed for reproducibility.
    """
    mode: NoiseMode = "both"
    cutoff_mhz: float = 2000.0  # MHz
    gain_dB: float = 5.0
    strength_amp: float =  DetectionConfig.sigma_vac*10**(gain_dB/20) # DetectionConfig.sigma_vac*np.sqrt((10 ** (gain_dB / 10.0) - 1))
    strength_phase: float =  DetectionConfig.sigma_vac*10**(gain_dB/20) # DetectionConfig.sigma_vac*np.sqrt((10 ** (gain_dB / 10.0) - 1))
    seed: int = 12345


@dataclass
class SpectrumConfig:
    """PSD estimation settings."""
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


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def next_pow_two(n: int) -> int:
    return 1 if n <= 1 else 2 ** (int(np.ceil(np.log2(n))))


def time_axis(duration_ps: float, dt_ps: float) -> Array:
    n = int(np.round(duration_ps / dt_ps))
    return np.arange(n, dtype=np.float64) * dt_ps


def frequency_axis(n: int, dt_ps: float) -> Array:
    return np.fft.fftfreq(n, d=dt_ps) * MHZ_PER_INV_PS


def complex_to_quadratures(z: ComplexArray) -> Tuple[Array, Array]:
    x = np.sqrt(2.0) * np.real(z)
    p = np.sqrt(2.0) * np.imag(z)
    return x, p


def quadrature_projection(z: ComplexArray, theta: float) -> Array:
    """Return the slow quadrature X_theta = sqrt(2) Re[e^{-i theta} z]."""
    return np.sqrt(2.0) * np.real(np.exp(-1j * theta) * z)


# -----------------------------------------------------------------------------
# Noise generation in Fourier space
# -----------------------------------------------------------------------------

def generate_band_limited_real_gaussian_noise(
    n: int,
    dt_ps: float,
    cutoff_mhz: float,
    rms: float,
    rng: np.random.Generator,
) -> Array:
    """Generate a real, zero-mean, approximately band-limited Gaussian process.

    Construction:
    - draw complex Gaussian Fourier coefficients
    - enforce Hermitian symmetry for a real time trace
    - keep only frequencies |f| <= cutoff_hz
    - inverse FFT
    - normalize to requested RMS

    The resulting process has a flat spectrum in the passband up to finite-size
    fluctuations and FFT normalization conventions.
    """
    if rms == 0.0:
        return np.zeros(n, dtype=np.float64)

    freqs = np.fft.rfftfreq(n, d=dt_ps) * MHZ_PER_INV_PS

    # Complex Gaussian coefficients on positive frequencies
    coeff = (rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size))
    mask = (freqs <= cutoff_mhz).astype(np.float64)
    coeff *= mask

    # Enforce real-valued time series conventions for rfft/irfft
    coeff[0] = coeff[0].real + 0j
    if n % 2 == 0:
        coeff[-1] = coeff[-1].real + 0j

    x = np.fft.irfft(coeff, n=n)
    x -= np.mean(x)

    current_rms = np.std(x)
    if current_rms > 0:
        x *= rms / current_rms
    return x


def generate_drive_noise(
    t: Array,
    cfg: NoiseConfig,
) -> Tuple[ComplexArray, Dict[str, Array]]:
    """Generate complex drive noise F_n(t).

    Returns
    -------
    F_n : complex ndarray
        Noise contribution added to the coherent drive F_s.
    aux : dict
        Dictionary with raw amplitude and phase noise traces and metadata.
    """
    n = t.size
    dt_ps = t[1] - t[0] if n > 1 else 1.0
    rng = np.random.default_rng(cfg.seed)

    amp = np.zeros(n, dtype=np.float64)
    ph = np.zeros(n, dtype=np.float64)

    if cfg.mode in ("amplitude", "both") and cfg.strength_amp != 0.0:
        amp = generate_band_limited_real_gaussian_noise(
            n=n,
            dt_ps=dt_ps,
            cutoff_mhz=cfg.cutoff_mhz,
            rms=cfg.strength_amp,
            rng=rng,
        )

    if cfg.mode in ("phase", "both") and cfg.strength_phase != 0.0:
        ph = generate_band_limited_real_gaussian_noise(
            n=n,
            dt_ps=dt_ps,
            cutoff_mhz=cfg.cutoff_mhz,
            rms=cfg.strength_phase,
            rng=rng,
        )

    F_n = (amp + 1j * ph) / np.sqrt(2)
    return F_n.astype(np.complex128), {
        "amp_noise": amp,
        "phase_noise": ph,
        "cutoff_mhz": np.array([cfg.cutoff_mhz], dtype=np.float64),
    }


# -----------------------------------------------------------------------------
# Microcavity dynamics
# -----------------------------------------------------------------------------

def cavity_rhs(
    psi: complex,
    F: complex,
    detuning_inv_ps: float,
    nonlinearity_inv_ps: float,
    kappa_out_inv_ps: float,
    loss_inv_ps: float,
    cfg: CavityConfig,
) -> complex:
    """Right-hand side of dψ/dt.

    Starting from
        i dψ/dt = [ -Δ + U |ψ|^2 - i γ/2 ] ψ + ik^0.5*F
    we get
        dψ/dt = iΔ ψ - iU |ψ|^2 ψ - (γ/2) ψ + k^0.5*F
    """
    return (
        1j * detuning_inv_ps * psi
        - 1j * nonlinearity_inv_ps * (abs(psi) ** 2) * psi
        - 0.5 * loss_inv_ps * psi
        + np.sqrt(cfg.kappa_out_inv_ps) * F
    )


def integrate_cavity(
    t: Array,
    F_t: ComplexArray,
    cfg: CavityConfig,
    integrator: str = "rk4",
) -> ComplexArray:
    """Integrate the single-mode cavity dynamics for a time-dependent drive F(t)."""
    n = t.size
    psi = np.empty(n, dtype=np.complex128)
    psi[0] = cfg.psi0
    dt = t[1] - t[0] if n > 1 else 1.0

    def rhs(state: complex, drive: complex) -> complex:
        return cavity_rhs(
            state,
            drive,
            detuning_inv_ps=cfg.detuning_inv_ps,
            nonlinearity_inv_ps=cfg.nonlinearity_inv_ps,
            kappa_out_inv_ps=cfg.kappa_out_inv_ps,
            loss_inv_ps=cfg.loss_inv_ps,
            cfg=cfg,
        )

    for k in range(n - 1):
        y = psi[k]
        f0 = F_t[k]
        f1 = F_t[k + 1]
        fmid = 0.5 * (f0 + f1)

        if integrator == "euler":
            psi[k + 1] = y + dt * rhs(y, f0)
        elif integrator == "heun":
            y_pred = y + dt * rhs(y, f0)
            psi[k + 1] = y + 0.5 * dt * (rhs(y, f0) + rhs(y_pred, f1))
        elif integrator == "rk4":
            k1 = rhs(y, f0)
            k2 = rhs(y + 0.5 * dt * k1, fmid)
            k3 = rhs(y + 0.5 * dt * k2, fmid)
            k4 = rhs(y + dt * k3, f1)
            psi[k + 1] = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

    return psi

# -----------------------------------------------------------------------------
# Noise utilities for detection
# -----------------------------------------------------------------------------


def add_white_noise_from_psd(
        signal: Array,
        noise_psd_per_mhz: float,
        dt_ps: float,
        rng: np.random.Generator,
) -> Array:
    """Add white Gaussian noise to a signal with a specified PSD."""
    fs_mhz = 1.0 / dt_ps * MHZ_PER_INV_PS
    sigma = math.sqrt(max(noise_psd_per_mhz, 0.0) * fs_mhz / 2.0)  # Here, sigma is a float
    noise = rng.normal(scale=sigma, size=signal.size)
    return signal + noise


def add_white_noise_from_local_psd(
        signal: Array,
        local_psd_per_mhz: Array,
        dt_ps: float,
        rng: np.random.Generator,
) -> Array:
    """Add white Gaussian noise with a local PSD that depends on the instantaneous signal value."""
    fs_mhz = 1.0 / dt_ps * MHZ_PER_INV_PS
    sigma = np.sqrt(np.maximum(local_psd_per_mhz, 0.0) * fs_mhz / 2.0) # Now sigma is an array of the same shape as signal
    noise = rng.normal(size=signal.size) * sigma
    return signal + noise


def shot_noise_psd_from_photocurrent(
        photocurrent: Array,
        cfg: DetectionConfig,
) -> Array:
    """Convert photocurrent to an effective shot-noise PSD in simulation units.
    
    Model:
    PSD_shot_noise = shot_noise_gain_per_current * photocurrent
    S_shot(MHz) = G_shot * I_photocurrent
    """
    return cfg.shot_noise_gain_per_current * np.maximum(photocurrent, 0.0)


# -----------------------------------------------------------------------------
# Input-output and detection
# -----------------------------------------------------------------------------


def output_field(F_t: ComplexArray, psi_t: ComplexArray, cfg: CavityConfig) -> ComplexArray:
    """Input-output relation for the reflected/transmitted field.

    Convention used here:
        s_out = F_in - sqrt(kappa_out) * psi

    Depending on your cavity geometry and normalization, you may prefer another
    convention (for example with + sign, or using a different coupling constant).
    This function is the only place you need to modify for that.
    """
    return F_t - np.sqrt(cfg.kappa_out_inv_ps) * psi_t


def balanced_homodyne_current(
    s_out_t: ComplexArray,
    cfg: DetectionConfig,
    rng: Optional[np.random.Generator] = None,
    dt_ps: Optional[float] = None,
) -> Dict[str, Array]:
    """Compute the balanced homodyne photocurrent time trace.

    Returns
    -------
    i_det : ndarray
        Deterministic homodyne current (without added detector shot noise).
    i_meas : ndarray
        Measured current including optional white shot-noise floor.
    """
    if rng is None:
        rng = np.random.default_rng(2026)
    if dt_ps is None and cfg.add_shot_noise and cfg.shot_noise_mode == "photocurrent":
        raise ValueError("dt_ps must be provided when add_shot_noise=True")

    scale = cfg.responsivity * cfg.detection_efficiency
    beta = cfg.lo_amplitude * np.exp(1j * cfg.lo_phase_rad)

    b1 = (s_out_t + beta) / np.sqrt(2.0)
    b2 = (s_out_t - beta) / np.sqrt(2.0)

    i1_det = scale * np.abs(b1) ** 2
    i2_det = scale * np.abs(b2) ** 2
    i_det = i1_det - i2_det

    i_meas = i_det.copy()

    if cfg.add_shot_noise:
        if cfg.shot_noise_mode == "fixed":
            i_meas = add_white_noise_from_psd(
                i_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
        elif cfg.shot_noise_mode == "photocurrent":
            shot_ref = i1_det + i2_det
            if cfg.shot_noise_use_instantaneous_photocurrent:
                local_psd = shot_noise_psd_from_photocurrent(shot_ref, cfg)
                i_meas = add_white_noise_from_local_psd(
                    i_meas,
                    local_psd_per_mhz=local_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
            else:
                mean_psd = float(np.mean(shot_noise_psd_from_photocurrent(shot_ref, cfg)))
                i_meas = add_white_noise_from_psd(
                    i_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
        elif cfg.shot_noise_mode != "none":
            raise ValueError(f"Unknown shot noise mode: {cfg.shot_noise_mode}")

        if cfg.electronic_noise_psd_per_mhz > 0.0:
            i_meas = add_white_noise_from_psd(
                i_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )

    return {
        "beta_lo_t": np.full(s_out_t.shape, beta, dtype=np.complex128),
        "b1_t": b1,
        "b2_t": b2,
        "i_1_det_t": i1_det,
        "i_2_det_t": i2_det,
        "i_det_t": i_det,
        "i_meas_t": i_meas
    }


def generate_vacuum_field(
    n: int,
    dt_ps: float,
    sigma_vac: float, 
    rng: np.random.Generator,
) -> ComplexArray:
    """Generate a complex vacuum noise field with a specified PSD.
    This model is useful to produce an realistic ESA trace, but it does not provide
    a rigorous quantum description of the vacuum fluctuations entering the unused port of the beamsplitter.
    """

    v_re = rng.normal(scale=sigma_vac, size=n)
    v_im = rng.normal(scale=sigma_vac, size=n)
    return ((v_re + 1j * v_im) / np.sqrt(2.0)).astype(np.complex128)


def balanced_current_for_drive_noise(
    F_t: ComplexArray,
    cfg: DetectionConfig,
    rng: Optional[np.random.Generator] = None,
    dt_ps: Optional[float] = None,
) -> Dict[str, Array]:
    """Compute the balanced homodyne current for the drive F_n(t) not going through the cavity."""
    if cfg.simulate_vacuum_port:
        v_t = generate_vacuum_field(
            n=F_t.size,
            dt_ps=dt_ps,
            sigma_vac=cfg.sigma_vac,
            rng=rng,
        )
    else:
        v_t = np.zeros_like(F_t)

    b1 = (F_t + v_t) / np.sqrt(2.0)
    b2 = (F_t - v_t) / np.sqrt(2.0)

    scale = cfg.responsivity * cfg.detection_efficiency

    i1_det_ref = scale * np.abs(b1) ** 2
    i2_det_ref = scale * np.abs(b2) ** 2

    i_plus_det_ref = i1_det_ref + i2_det_ref  
    i_minus_det_ref = i1_det_ref - i2_det_ref

    return {
        "b1_ref_t": b1,
        "b2_ref_t": b2,
        "i_1_det_ref_t": i1_det_ref,
        "i_2_det_ref_t": i2_det_ref,
        "i_plus_det_ref_t": i_plus_det_ref,
        "i_minus_det_ref_t": i_minus_det_ref,
    }


def balanced_direct_detection_currents_without_noise(
        s_out_without_noise_t: ComplexArray,
        cfg: DetectionConfig,
        rng: Optional[np.random.Generator] = None,
        dt_ps: Optional[float] = None,
) -> Dict[str, Array]:
    """Balanced detection after BS 50/50 without LO phase control, with no noise added
    Field on the photodiode :
        b1 = (s_out_wn + v) / sqrt(2)
        b2 = (s_out_wn - v) / sqrt(2)
    
    Currents :
        i1 = R * |b1|^2
        i2 = R * |b2|^2
    where R is the responsivity. 
    
    The sum and difference currents are then:
        i_sum = i1 + i2 = R * (|s_out_wn|^2 + |v|^2)
        i_diff = i1 - i2 = R * 2 Re[s_out_wn v*]
    """
    if rng is None:
        rng = np.random.default_rng(2027)
    if dt_ps is None:
        raise ValueError("dt_ps must be provided for balanced direct detection")
    
    scale = cfg.responsivity * cfg.detection_efficiency

    if cfg.simulate_vacuum_port:
        v_t = generate_vacuum_field(
            n=s_out_without_noise_t.size,
            dt_ps=dt_ps,
            sigma_vac=cfg.sigma_vac,
            rng=rng,
        )
    else:
        v_t = np.zeros_like(s_out_without_noise_t)

    b1 = (s_out_without_noise_t + v_t) / np.sqrt(2.0)
    b2 = (s_out_without_noise_t - v_t) / np.sqrt(2.0)

    i1_det = scale * np.abs(b1) ** 2
    i2_det = scale * np.abs(b2) ** 2

    i_plus_det = i1_det + i2_det
    i_minus_det = i1_det - i2_det

    i1_meas = i1_det.copy()
    i2_meas = i2_det.copy()
    i_plus_meas = i_plus_det.copy()
    i_minus_meas = i_minus_det.copy()

    #  If we are simulating the vacuum port, the shot noise is already included
    # in the generated vacuum field, so we should not add additional noise on top of that.
    if cfg.add_shot_noise and not cfg.simulate_vacuum_port: 
        if cfg.shot_noise_mode == "fixed":
            i1_meas = add_white_noise_from_psd(
                i1_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i2_meas = add_white_noise_from_psd(
                i2_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_plus_meas = add_white_noise_from_psd(
                i_plus_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_minus_meas = add_white_noise_from_psd(
                i_minus_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )

        elif cfg.shot_noise_mode == "photocurrent":
            psd_i1 = shot_noise_psd_from_photocurrent(i1_det, cfg)
            psd_i2 = shot_noise_psd_from_photocurrent(i2_det, cfg)
            psd_sumdiff = shot_noise_psd_from_photocurrent(i_plus_det, cfg)  # Approximate shot noise PSD for sum/diff as function of total photocurrent

            if cfg.shot_noise_use_instantaneous_photocurrent:
                i1_meas = add_white_noise_from_local_psd(
                    i1_meas,
                    local_psd_per_mhz=psd_i1,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i2_meas = add_white_noise_from_local_psd(
                    i2_meas,
                    local_psd_per_mhz=psd_i2,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_plus_meas = add_white_noise_from_local_psd(
                    i_plus_meas,
                    local_psd_per_mhz=psd_sumdiff,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_minus_meas = add_white_noise_from_local_psd(
                    i_minus_meas,
                    local_psd_per_mhz=psd_sumdiff,
                    dt_ps=dt_ps,
                    rng=rng,
                )

            else:
                mean_psd = float(np.mean(shot_noise_psd_from_photocurrent(psd_sumdiff, cfg)))
                i1_meas = add_white_noise_from_psd(
                    i1_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i2_meas = add_white_noise_from_psd(
                    i2_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_plus_meas = add_white_noise_from_psd(
                    i_plus_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_minus_meas = add_white_noise_from_psd(
                    i_minus_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
        
        elif cfg.shot_noise_mode != "none":
            raise ValueError(f"Unknown shot noise mode: {cfg.shot_noise_mode}")
        
        if cfg.electronic_noise_psd_per_mhz > 0.0:
            i1_meas = add_white_noise_from_psd(
                i1_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i2_meas = add_white_noise_from_psd(
                i2_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_plus_meas = add_white_noise_from_psd(
                i_plus_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_minus_meas = add_white_noise_from_psd(
                i_minus_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )

    return {
        "vacuum_t_wn": v_t,
        "b1_t_wn": b1,
        "b2_t_wn": b2,
        "i1_det_t_wn": i1_det,
        "i2_det_t_wn": i2_det,
        "i_plus_det_t_wn": i_plus_det,
        "i_minus_det_t_wn": i_minus_det,
        "i1_meas_t_wn": i1_meas,
        "i2_meas_t_wn": i2_meas,
        "i_plus_meas_t_wn": i_plus_meas,
        "i_minus_meas_t_wn": i_minus_meas,
    }


def balanced_direct_detection_currents(
        s_out_t: ComplexArray,
        cfg: DetectionConfig,
        rng: Optional[np.random.Generator] = None,
        dt_ps: Optional[float] = None,
) -> Dict[str, Array]:
    """Balanced detection after BS 50/50 without LO phase control.
    Field on the photodiode :
        b1 = (s_out + v) / sqrt(2)
        b2 = (s_out - v) / sqrt(2)
    
    Currents :
        i1 = R * |b1|^2
        i2 = R * |b2|^2
    where R is the responsivity. 
    
    The sum and difference currents are then:
        i_sum = i1 + i2 = R * (|s_out|^2 + |v|^2)
        i_diff = i1 - i2 = R * 2 Re[s_out v*]
    """
    if rng is None:
        rng = np.random.default_rng(2027)
    if dt_ps is None:
        raise ValueError("dt_ps must be provided for balanced direct detection")
    
    scale = cfg.responsivity * cfg.detection_efficiency

    if cfg.simulate_vacuum_port:
        v_t = generate_vacuum_field(
            n=s_out_t.size,
            dt_ps=dt_ps,
            sigma_vac=cfg.sigma_vac,
            rng=rng,
        )
    else:
        v_t = np.zeros_like(s_out_t)

    b1 = (s_out_t + v_t) / np.sqrt(2.0)
    b2 = (s_out_t - v_t) / np.sqrt(2.0)

    i1_det = scale * np.abs(b1) ** 2
    i2_det = scale * np.abs(b2) ** 2

    i_plus_det = i1_det + i2_det
    i_minus_det = i1_det - i2_det

    i1_meas = i1_det.copy()
    i2_meas = i2_det.copy()
    i_plus_meas = i_plus_det.copy()
    i_minus_meas = i_minus_det.copy()

    #  If we are simulating the vacuum port, the shot noise is already included
    # in the generated vacuum field, so we should not add additional noise on top of that.
    if cfg.add_shot_noise and not cfg.simulate_vacuum_port: 
        if cfg.shot_noise_mode == "fixed":
            i1_meas = add_white_noise_from_psd(
                i1_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i2_meas = add_white_noise_from_psd(
                i2_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_plus_meas = add_white_noise_from_psd(
                i_plus_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_minus_meas = add_white_noise_from_psd(
                i_minus_meas,
                noise_psd_per_mhz=cfg.shot_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )

        elif cfg.shot_noise_mode == "photocurrent":
            psd_i1 = shot_noise_psd_from_photocurrent(i1_det, cfg)
            psd_i2 = shot_noise_psd_from_photocurrent(i2_det, cfg)
            psd_sumdiff = shot_noise_psd_from_photocurrent(i_plus_det, cfg)  # Approximate shot noise PSD for sum/diff as function of total photocurrent

            if cfg.shot_noise_use_instantaneous_photocurrent:
                i1_meas = add_white_noise_from_local_psd(
                    i1_meas,
                    local_psd_per_mhz=psd_i1,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i2_meas = add_white_noise_from_local_psd(
                    i2_meas,
                    local_psd_per_mhz=psd_i2,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_plus_meas = add_white_noise_from_local_psd(
                    i_plus_meas,
                    local_psd_per_mhz=psd_sumdiff,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_minus_meas = add_white_noise_from_local_psd(
                    i_minus_meas,
                    local_psd_per_mhz=psd_sumdiff,
                    dt_ps=dt_ps,
                    rng=rng,
                )

            else:
                mean_psd = float(np.mean(shot_noise_psd_from_photocurrent(psd_sumdiff, cfg)))
                i1_meas = add_white_noise_from_psd(
                    i1_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i2_meas = add_white_noise_from_psd(
                    i2_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_plus_meas = add_white_noise_from_psd(
                    i_plus_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
                i_minus_meas = add_white_noise_from_psd(
                    i_minus_meas,
                    noise_psd_per_mhz=mean_psd,
                    dt_ps=dt_ps,
                    rng=rng,
                )
        
        elif cfg.shot_noise_mode != "none":
            raise ValueError(f"Unknown shot noise mode: {cfg.shot_noise_mode}")
        
        if cfg.electronic_noise_psd_per_mhz > 0.0:
            i1_meas = add_white_noise_from_psd(
                i1_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i2_meas = add_white_noise_from_psd(
                i2_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_plus_meas = add_white_noise_from_psd(
                i_plus_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )
            i_minus_meas = add_white_noise_from_psd(
                i_minus_meas,
                noise_psd_per_mhz=cfg.electronic_noise_psd_per_mhz,
                dt_ps=dt_ps,
                rng=rng,
            )

    return {
        "vacuum_t": v_t,
        "b1_t": b1,
        "b2_t": b2,
        "i1_det_t": i1_det,
        "i2_det_t": i2_det,
        "i_plus_det_t": i_plus_det,
        "i_minus_det_t": i_minus_det,
        "i1_meas_t": i1_meas,
        "i2_meas_t": i2_meas,
        "i_plus_meas_t": i_plus_meas,
        "i_minus_meas_t": i_minus_meas,
    }


# -----------------------------------------------------------------------------
# PSD estimation / spectrum analyzer emulation
# -----------------------------------------------------------------------------

def compute_psd(
    x: Array,
    fs_mhz: float,
    cfg: SpectrumConfig,
) -> Tuple[Array, Array]:
    """Compute one-sided PSD using Welch if available, otherwise an FFT fallback."""
    if SCIPY_AVAILABLE:
        f, pxx = welch(
            x,
            fs=fs_mhz,
            window=cfg.window,
            nperseg=min(cfg.nperseg, x.size),
            detrend=cfg.detrend,
            return_onesided=True,
            scaling="density",
            average=cfg.average,
        )
        return f.astype(np.float64), pxx.astype(np.float64)

    # Fallback: single-shot FFT periodogram with Hann window.
    n = x.size
    if n < 2:
        raise ValueError("Need at least 2 points to compute a PSD.")
    window = np.hanning(n)
    xw = (x - np.mean(x)) * window
    norm = fs_mhz * np.sum(window ** 2)
    Xf = np.fft.rfft(xw)
    pxx = (np.abs(Xf) ** 2) / norm
    f = np.fft.rfftfreq(n, d=1.0 / fs_mhz)
    return f.astype(np.float64), pxx.astype(np.float64)


def rbw_average_psd(
    freqs_mhz: Array,
    psd: Array,
    rbw_mhz: float,
) -> Tuple[Array, Array]:
    """Mimic a simple spectrum-analyzer resolution bandwidth by box averaging.

    This is optional and approximate. It helps produce visually realistic traces.
    """
    if rbw_mhz <= 0:
        return freqs_mhz, psd

    df = freqs_mhz[1] - freqs_mhz[0]
    bins = max(1, int(round(rbw_mhz / df)))
    if bins == 1:
        return freqs_mhz, psd

    n = (psd.size // bins) * bins
    f2 = freqs_mhz[:n].reshape(-1, bins).mean(axis=1)
    p2 = psd[:n].reshape(-1, bins).mean(axis=1)
    return f2, p2

# -----------------------------------------------------------------------------
# Bistability preparation / pump sweep
# -----------------------------------------------------------------------------

def compute_bistability_curve(
    cfg: FullConfig,
    F_values: Array,
    settle_time_ps: float = 2e4,
) -> Dict[str, Array]:
    """Compute hysteresis curve by sweeping pump amplitude up and down."""

    t = time_axis(settle_time_ps, cfg.sim.dt_ps)

    density_up = []
    density_down = []

    old_psi0 = cfg.cavity.psi0

    # Sweep up
    psi0 = 0.0 + 0.0j

    for F in F_values:
        cfg.cavity.psi0 = psi0
        F_t = np.full(t.shape, F + 0j, dtype=np.complex128)

        psi_t = integrate_cavity(
            t,
            F_t,
            cfg.cavity,
            integrator=cfg.sim.integrator,
        )

        psi0 = psi_t[-1]
        density_up.append(np.abs(psi0) ** 2)

    # Sweep down
    for F in F_values[::-1]:
        cfg.cavity.psi0 = psi0
        F_t = np.full(t.shape, F + 0j, dtype=np.complex128)

        psi_t = integrate_cavity(
            t,
            F_t,
            cfg.cavity,
            integrator=cfg.sim.integrator,
        )

        psi0 = psi_t[-1]
        density_down.append(np.abs(psi0) ** 2)

    cfg.cavity.psi0 = old_psi0

    return {
        "F_up": F_values,
        "density_up": np.array(density_up),
        "F_down": F_values[::-1],
        "density_down": np.array(density_down),
    }


def make_square_cycle_drive(
    t: Array,
    F_low: complex,
    F_high: complex,
    F_work: complex,
    t_rise_ps: float,
    t_fall_ps: float,
) -> ComplexArray:
    """Generate pump cycle: low -> high -> working point."""

    F_t = np.full(t.shape, F_low, dtype=np.complex128)

    F_t[t >= t_rise_ps] = F_high
    F_t[t >= t_fall_ps] = F_work

    return F_t


def prepare_upper_branch(
    cfg: FullConfig,
    F_low: complex,
    F_high: complex,
    F_work: complex,
) -> complex:
    """Prepare the cavity field on the upper bistable branch."""

    t_prep = time_axis(cfg.sim.duration_ps, cfg.sim.dt_ps)

    F_prep_t = make_square_cycle_drive(
        t=t_prep,
        F_low=F_low,
        F_high=F_high,
        F_work=F_work,
        t_rise_ps=0.2 * cfg.sim.duration_ps,
        t_fall_ps=0.7 * cfg.sim.duration_ps,
    )

    old_psi0 = cfg.cavity.psi0
    cfg.cavity.psi0 = 0.0 + 0.0j

    psi_prep_t = integrate_cavity(
        t=t_prep,
        F_t=F_prep_t,
        cfg=cfg.cavity,
        integrator=cfg.sim.integrator,
    )

    cfg.cavity.psi0 = old_psi0

    psi_upper = psi_prep_t[-1]

    print("Prepared upper-branch density:", np.abs(psi_upper) ** 2)

    return psi_upper

# -----------------------------------------------------------------------------
# High-level simulation pipeline
# -----------------------------------------------------------------------------

#def run_simulation(cfg: FullConfig) -> Dict[str, np.ndarray]:
def run_simulation_with_upper_branch(
    cfg: FullConfig,
    F_low: complex,
    F_high: complex,
    F_work: complex,
) -> Dict[str, np.ndarray]:
    """Run the full pipeline from noisy optical drive to homodyne PSD."""
    t_full = time_axis(cfg.sim.duration_ps, cfg.sim.dt_ps)
    dt_ps = cfg.sim.dt_ps
    fs_mhz = 1.0 / (dt_ps) * MHZ_PER_INV_PS  # Convert to MHz

    # Generate noisy input drive
    #F_n, noise_aux = generate_drive_noise(t_full, cfg.noise)
    #F_t = cfg.cavity.F_s + F_n

    # Integrate cavity dynamics
    #psi_t = integrate_cavity(t_full, F_t, cfg.cavity, integrator=cfg.sim.integrator)
    #psi_t_wn = integrate_cavity(t_full, np.full(t_full.size, cfg.cavity.F_s), cfg.cavity, integrator=cfg.sim.integrator)

    # Prepare upper branch
    psi_upper = prepare_upper_branch(
        cfg,
        F_low=F_low,
        F_high=F_high,
        F_work=F_work,
    )

    # Use upper branch as initial condition
    cfg.cavity.psi0 = psi_upper
    cfg.cavity.F_s = F_work

    # Generate noisy input drive around working point
    F_n, noise_aux = generate_drive_noise(t_full, cfg.noise)
    F_t = cfg.cavity.F_s + F_n

    # Integrate noisy dynamics from upper branch
    psi_t = integrate_cavity(
        t_full,
        F_t,
        cfg.cavity,
        integrator=cfg.sim.integrator,
    )

    # Reference without added noise, also from upper branch
    psi_t_wn = integrate_cavity(
        t_full,
        np.full(t_full.size, cfg.cavity.F_s),
        cfg.cavity,
        integrator=cfg.sim.integrator,
    )
        
    # Compute output field
    s_out_t = output_field(F_t, psi_t, cfg.cavity)
    s_out_without_noise_t = output_field(cfg.cavity.F_s, psi_t_wn, cfg.cavity)

    rng = np.random.default_rng(cfg.noise.seed + 999)

    results: Dict[str, np.ndarray] = {
        "t_ps": t_full,
        "F_t": F_t,
        "psi_t": psi_t,
        "s_out_t": s_out_t,
        "s_out_without_noise_t": s_out_without_noise_t,
        "amp_noise": noise_aux["amp_noise"],
        "phase_noise": noise_aux["phase_noise"],
    }

    # Choosed detection scheme
    if cfg.detection.mode == "homodyne":
        det = balanced_homodyne_current(
            s_out_t=s_out_t,
            cfg=cfg.detection,
            rng=rng,
            dt_ps=dt_ps,
            )
        results.update(det)

        signal_for_psd_det = det["i_det_t"]
        signal_for_psd_meas = det["i_meas_t"]

    elif cfg.detection.mode in ("balanced_sum", "balanced_diff"):
        det = balanced_direct_detection_currents(
            s_out_t=s_out_t,
            cfg=cfg.detection,
            rng=rng,
            dt_ps=dt_ps,
        )
        results.update(det)

        det_without_noise = balanced_direct_detection_currents_without_noise(
            s_out_without_noise_t=s_out_without_noise_t,
            cfg=cfg.detection,
            rng=rng,
            dt_ps=dt_ps,
        )
        results.update(det_without_noise)

        ref = balanced_current_for_drive_noise(
            F_t=F_t,
            cfg=cfg.detection,
            rng=rng,
            dt_ps=dt_ps,
        )

        results.update(ref)

        if cfg.detection.mode == "balanced_sum":
            signal_for_psd_det = det["i_plus_det_t"]
            signal_for_psd_meas = det["i_plus_meas_t"]
            signal_for_psd_ref = ref["i_plus_det_ref_t"]
            signal_for_psd_det_without_noise = det_without_noise["i_plus_det_t_wn"]
        else:
            signal_for_psd_det = det["i_minus_det_t"]
            signal_for_psd_meas = det["i_minus_meas_t"]
            signal_for_psd_ref = ref["i_minus_det_ref_t"]
            signal_for_psd_det_without_noise = det_without_noise["i_minus_det_t_wn"]
        
        results["i_det_t"] = signal_for_psd_det
        results["i_meas_t"] = signal_for_psd_meas
        results["i_ref_t"] = signal_for_psd_ref
        results["i_det_without_noise_t"] = signal_for_psd_det_without_noise
  
    else:
        raise ValueError(f"Unknown detection mode: {cfg.detection.mode}")

    # Discard transient and optionally downsample stored arrays
    n0 = int(cfg.sim.discard_fraction * t_full.size)
    step = max(1, cfg.sim.store_every)

    for k in list(results.keys()):
        results[k] = results[k][n0::step]
        if isinstance(results[k], np.ndarray) and results[k].shape == t_full.shape:
            results[k] = results[k][n0::step]

    fs_store_mhz = fs_mhz / step

    # PSDs for the spectrum analyzer trace
    f_det, psd_det = compute_psd(results["i_det_t"], fs_store_mhz, cfg.spectrum)
    f_meas, psd_meas = compute_psd(results["i_meas_t"], fs_store_mhz, cfg.spectrum)

    # PSDs for the drive noise alone (for diagnostics)
    f_drive, psd_drive = compute_psd(results["i_ref_t"], fs_store_mhz, cfg.spectrum)

    # PSDs for the spectrum reference (without noise)
    f_wn, psd_wn = compute_psd(results["i_det_without_noise_t"], fs_store_mhz, cfg.spectrum)

    # Also compute cavity/output quadratures for diagnostics
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
        "freqs_det_mhz": f_det,
        "psd_det": psd_det,
        "freqs_meas_mhz": f_meas,
        "psd_meas": psd_meas,
        "freqs_drive_mhz": f_drive,
        "psd_drive": psd_drive,
        "freqs_wn_mhz": f_wn,
        "psd_wn": psd_wn,
        "fs_store_mhz": np.array([fs_store_mhz], dtype=np.float64),
    })

    return results


# -----------------------------------------------------------------------------
# Diagnostics and plotting
# -----------------------------------------------------------------------------

def estimate_quadrature_variances(z: ComplexArray) -> Dict[str, float]:
    x, p = complex_to_quadratures(z)
    return {
        "var_x": float(np.var(x)),
        "var_p": float(np.var(p)),
        "cov_xp": float(np.cov(x, p, ddof=0)[0, 1]),
        "mean_x": float(np.mean(x)),
        "mean_p": float(np.mean(p)),
    }


def plot_time_traces(results: Dict[str, np.ndarray], max_points: int = 5000) -> None:
    t = results["t_ps"]
    n = t.size
    step = max(1, n // max_points)
    sl = slice(None, None, step)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(t[sl] * 1e6, results["amp_noise"][sl], label="Amplitude noise")
    axes[0].plot(t[sl] * 1e6, results["phase_noise"][sl], label="Phase noise")
    axes[0].set_ylabel("Drive noise")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t[sl] * 1e6, np.real(results["F_t"][sl]), label="Re F(t)")
    axes[1].plot(t[sl] * 1e6, np.imag(results["F_t"][sl]), label="Im F(t)")
    axes[1].set_ylabel("Input drive")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t[sl] * 1e6, np.real(results["psi_t"][sl]), label="Re ψ(t)")
    axes[2].plot(t[sl] * 1e6, np.imag(results["psi_t"][sl]), label="Im ψ(t)")
    axes[2].set_ylabel("Intracavity field")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    if "i_plus_meas_t" in results:
        axes[3].plot(t[sl] * 1e6, results["i_plus_meas_t"][sl], label="Sum current")
    elif "i_minus_meas_t" in results:
        axes[3].plot(t[sl] * 1e6, results["i_minus_meas_t"][sl], label="Difference current")
    else:
        axes[3].plot(t[sl] * 1e6, results["i_meas_t"][sl], label="Homodyne current")

    axes[3].set_ylabel("Photocurrent")
    axes[3].set_xlabel("Time (ps)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_phase_space(results: Dict[str, np.ndarray], max_points: int = 50000) -> None:
    x_in, p_in = results["x_in"], results["p_in"]
    x_out, p_out = results["x_out"], results["p_out"]

    n = x_in.size
    step = max(1, n // max_points)
    sl = slice(None, None, step)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(x_in[sl], p_in[sl], s=2, alpha=0.2)
    axes[0].set_title("Input drive quadratures")
    axes[0].set_xlabel("X_in")
    axes[0].set_ylabel("P_in")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(x_out[sl], p_out[sl], s=2, alpha=0.2)
    axes[1].set_title("Output field quadratures")
    axes[1].set_xlabel("X_out")
    axes[1].set_ylabel("P_out")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_spectra(
    results: Dict[str, np.ndarray],
    rbw_mhz: float = 0.0,
    fmin_mhz: Optional[float] = None,
    fmax_mhz: Optional[float] = None,
    loglog: bool = False,
) -> None:
    f1 = results["freqs_det_mhz"]
    p1 = results["psd_det"]
    f2 = results["freqs_meas_mhz"]
    p2 = results["psd_meas"]

    if rbw_mhz > 0:
        f1, p1 = rbw_average_psd(f1, p1, rbw_mhz)
        f2, p2 = rbw_average_psd(f2, p2, rbw_mhz)

    mask1 = np.ones_like(f1, dtype=bool)
    mask2 = np.ones_like(f2, dtype=bool)
    if fmin_mhz is not None:
        mask1 &= (f1 >= fmin_mhz)
        mask2 &= (f2 >= fmin_mhz)
    if fmax_mhz is not None:
        mask1 &= (f1 <= fmax_mhz)
        mask2 &= (f2 <= fmax_mhz)

    plt.figure(figsize=(10, 6))
    if loglog:
        plt.loglog(f1[mask1], p1[mask1], label="Deterministic homodyne PSD")
        plt.loglog(f2[mask2], p2[mask2], label="Measured PSD (with shot noise)")
    else:
        plt.semilogy(f1[mask1], p1[mask1], label="Deterministic homodyne PSD")
        plt.semilogy(f2[mask2], p2[mask2], label="Measured PSD (with shot noise)")
    plt.xlabel("Analysis frequency (MHz)")
    plt.ylabel("PSD (current units$^2$/MHz)")
    plt.title("Spectrum analyzer trace after balanced homodyne detection")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Parameter sweeps
# -----------------------------------------------------------------------------

def clone_config(base_cfg: FullConfig) -> FullConfig:
    """Create a deep copy of the base config to modify for parameter sweeps."""
    return FullConfig(
        noise=NoiseConfig(**asdict(base_cfg.noise)),
        cavity=CavityConfig(**asdict(base_cfg.cavity)),
        sim=SimulationConfig(**asdict(base_cfg.sim)),
        detection=DetectionConfig(**asdict(base_cfg.detection)),
        spectrum=SpectrumConfig(**asdict(base_cfg.spectrum)),
    )


def sweep_lo_phase(
    base_cfg: FullConfig,
    phases_rad: Array,
    F_low: complex,
    F_high: complex,
    F_work: complex,
) -> Dict[str, Array]:
    """Sweep homodyne phase and record detected variance."""

    vals = []

    for theta in phases_rad:
        cfg = clone_config(base_cfg)

        cfg.detection.mode = "homodyne"
        cfg.detection.lo_phase_rad = float(theta)

        res = run_simulation_with_upper_branch(
            cfg,
            F_low=F_low,
            F_high=F_high,
            F_work=F_work,
        )

        vals.append(np.var(res["i_det_t"]))

    return {
        "phases_rad": np.asarray(phases_rad),
        "var_i_det": np.asarray(vals),
    }


def sweep_noise_strength(
    base_cfg: FullConfig,
    strengths: Array,
    F_low: complex,
    F_high: complex,
    F_work: complex,
    which: Literal["amplitude", "phase"] = "phase",
) -> Dict[str, Array]:
    """Sweep noise level and record integrated output PSD."""

    integrated = []

    for s in strengths:
        cfg = clone_config(base_cfg)

        if which == "phase":
            cfg.noise.mode = "phase"
            cfg.noise.strength_phase = float(s)
            cfg.noise.strength_amp = 0.0
        else:
            cfg.noise.mode = "amplitude"
            cfg.noise.strength_amp = float(s)
            cfg.noise.strength_phase = 0.0

        res = run_simulation_with_upper_branch(
            cfg,
            F_low=F_low,
            F_high=F_high,
            F_work=F_work,
        )

        f = res["freqs_det_mhz"]
        p = res["psd_det"]

        integrated.append(np.trapz(p, f))

    return {
        "strengths": np.asarray(strengths),
        "integrated_psd": np.asarray(integrated),
    }


# -----------------------------------------------------------------------------
# Saving and loading
# -----------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert config objects to JSON-serializable objects.

    In particular, Python complex numbers (for example F_s or psi0) are not
    directly serializable by json.dumps, so we store them as tagged dicts.
    """
    if isinstance(obj, complex):
        return {
            "__complex__": True,
            "real": float(np.real(obj)),
            "imag": float(np.imag(obj)),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def save_results_npz(path: str, cfg: FullConfig, results: Dict[str, np.ndarray]) -> None:
    meta = {
        "noise": asdict(cfg.noise),
        "cavity": asdict(cfg.cavity),
        "sim": asdict(cfg.sim),
        "detection": asdict(cfg.detection),
        "spectrum": asdict(cfg.spectrum),
    }
    meta_json = json.dumps(_json_safe(meta), indent=2)
    np.savez_compressed(path, metadata_json=meta_json, **results)


# -----------------------------------------------------------------------------
# Example main program
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = FullConfig()

    # Choose detection scheme: "homodyne", "balanced_sum", or "balanced_diff".
    cfg.detection.mode = "balanced_sum"  

    # Shot-noise model
    cfg.detection.add_shot_noise = True
    cfg.detection.shot_noise_mode = "photocurrent"  # "fixed", "photocurrent", or "none"
    cfg.detection.shot_noise_gain_per_current = 1.0  # PSD per unit photocurrent (simulation units)
    cfg.detection.shot_noise_use_instantaneous_photocurrent = True  # Whether to use the instantaneous photocurrent to determine the local shot noise PSD, or a fixed PSD based on the mean photocurrent.
    
    # Optional stationary electronics foor
    cfg.detection.electronic_noise_psd_per_mhz = 0.0  # Add a fixed electronic noise floor to the measured current PSD (in current units^2/MHz)

    #results = run_simulation(cfg)

    # -------------------------------------------------
    # 1. Plot bistability curve
    # -------------------------------------------------

    F_values = np.linspace(0.0, 3.0, 100)

    bistab = compute_bistability_curve(
        cfg,
        F_values,
    )
    
    bistab_results = {
    "bistab_F_up": bistab["F_up"],
    "bistab_density_up": bistab["density_up"],
    "bistab_F_down": bistab["F_down"],
    "bistab_density_down": bistab["density_down"],
}

    plt.figure(figsize=(7, 5))
    plt.scatter(bistab["F_up"], bistab["density_up"], label="Sweep up", marker="x")
    plt.scatter(bistab["F_down"], bistab["density_down"], label="Sweep down", marker="+")
    plt.xlabel("Pump amplitude F")
    plt.ylabel(r"Intracavity density $|\psi|^2$")
    plt.title("Polariton bistability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------
    # 2. Choose pump values from the bistability curve
    # -------------------------------------------------

    F_low = 0.3 + 0j
    F_high = 1.2 + 0j
    F_work = 0.75 + 0j

    cfg.cavity.F_s = F_work



    # -------------------------------------------------
    # 3. Run noisy simulation on upper branch
    # -------------------------------------------------

    results = run_simulation_with_upper_branch(
        cfg,
        F_low=F_low,
        F_high=F_high,
        F_work=F_work,
    )

    results.update(bistab_results)

    results["F_low"] = np.array([F_low], dtype=np.complex128)
    results["F_high"] = np.array([F_high], dtype=np.complex128)
    results["F_work"] = np.array([F_work], dtype=np.complex128)

    # Diagnostics
    drive_stats = estimate_quadrature_variances(results["F_t"])
    output_stats = estimate_quadrature_variances(results["s_out_t"])
    print("Input drive quadrature stats:")
    for k, v in drive_stats.items():
        print(f"  {k:>10s} = {v:.6g}")
    print("\nOutput field quadrature stats:")
    for k, v in output_stats.items():
        print(f"  {k:>10s} = {v:.6g}")

    # Plots
    plot_time_traces(results)
    plot_phase_space(results)
    plot_spectra(results, rbw_mhz=10, fmin_mhz=1e3/1e6, fmax_mhz=2000)


    # Save
    #save_results_npz("/Users/charlotte/Documents/Thermal-states/polariton_homodyne_results_balanced_both_6.npz", cfg, results)
    save_results_npz("polariton_homodyne_results_balanced_both_7.npz", cfg, results)
    print("\nSaved results to polariton_homodyne_results_balanced_both_7.npz")


if __name__ == "__main__":
    main()

