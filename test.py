#%%
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional, Literal
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

HBAR_MEV_PS = 0.6582119514  # ħ in meV*ps
MHZ_PER_INV_PS = 1.0e6  # 1/(ps) in MHz
INV_PS_PER_MHZ = 1.0 / MHZ_PER_INV_PS

duration_ps: float = 1.0e6        # total simulated time in ps
dt_ps: float = 1              # timestep; sampling rate = 1/dt


def time_axis(duration_ps: float, dt_ps: float) -> Array:
    n = int(np.round(duration_ps / dt_ps))
    return np.arange(n, dtype=np.float64) * dt_ps


def frequency_axis(n: int, dt_ps: float) -> Array:
    return np.fft.fftfreq(n, d=dt_ps) * MHZ_PER_INV_PS

#%%
n = int(np.round(duration_ps / dt_ps))
print(n)
t = time_axis(duration_ps, dt_ps)
f = frequency_axis(n, dt_ps)
print(f)
print(1/dt_ps * MHZ_PER_INV_PS)
print(np.min(np.abs(f[1:])))
# %%
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

    freqs = np.fft.rfftfreq(n, d=dt_ps)

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

rms = 1.0
cutoff_mhz = 100
# %%
rng = np.random.default_rng(seed=42)
noise = generate_band_limited_real_gaussian_noise(n, dt_ps, cutoff_mhz, rms, rng)
# %%
plt.plot(t, noise)
plt.xlabel("Time (ps)")
plt.ylabel("Noise")
plt.title(f"Band-limited Gaussian noise (cutoff={cutoff_mhz} MHz, rms={rms})")
plt.show()

# %%
psd = np.abs(np.fft.rfft(noise))**2 * dt_ps / n 
plt.plot(f[:psd.size], psd)
plt.xlabel("Frequency (MHz)")
plt.ylabel("PSD (units^2/Hz)")
plt.title("Power Spectral Density of the Noise")
plt.xlim(0, 2*cutoff_mhz)
plt.show()

print(np.max(psd))
# %%
def add_white_electronic_noise(
        signal: Array,
        noise_psd_per_mhz: float,
        dt_ps: float,
        rng: np.random.Generator,
) -> Array:
    """Add white Gaussian noise to a signal with a specified PSD."""
    fs_mhz = 1.0 / dt_ps * MHZ_PER_INV_PS
    sigma = math.sqrt(max(noise_psd_per_mhz, 0.0) * fs_mhz / 2.0)
    noise = rng.normal(scale=sigma, size=signal.size)
    return signal + noise

signal = np.zeros(n)
noise_psd_per_mhz = 1
dt_ps = 1
rng = np.random.default_rng(seed=42)
noisy_signal = add_white_electronic_noise(signal, noise_psd_per_mhz, dt_ps, rng)
plt.plot(t, noisy_signal)
plt.xlabel("Time (ps)")
plt.ylabel("Noisy Signal")
plt.title(f"Noisy Signal with White Noise (PSD={noise_psd_per_mhz} units^2/MHz)")
plt.show() 
# %%
print(np.std(noisy_signal))
# %%
fs_mhz = 1.0 / dt_ps * MHZ_PER_INV_PS
expected_noise_std = math.sqrt(noise_psd_per_mhz * fs_mhz / 2.0)
print(expected_noise_std)

#%% 
import zipfile

path = "polariton_homodyne_results_balanced_both.npz"

with zipfile.ZipFile(path, "r") as zf:
    bad = zf.testzip()
    print("bad file:", bad)
    print(zf.namelist())
# %%
import numpy as np

data = np.load(path, allow_pickle=True)
print(data.files)
print(np.mean(np.imag(data["F_t"]))) 
# %%
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional, Literal
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.stats import gaussian_kde

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
    detuning_inv_ps: float = 1.4e-1 / HBAR_MEV_PS   # Δ = δ_meV / ħ
    nonlinearity_inv_ps: float =  0 ## 1.2e-2 / HBAR_MEV_PS  # U = g_meV_um2 / ħ
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
    mode: DetectionMode = "balanced_sum"

    # Homodyne parameters
    lo_amplitude: float = 50.0        # |β_LO| (arbitrary units)
    lo_phase_rad: float = np.pi / 2   # θ = π/2 detects phase quadrature

    # Vacuum port for balanced detection
    simulate_vacuum_port: bool = True
    sigma_vac: float = 0.02


@dataclass
class NoiseConfig:
    mode: NoiseMode = "amplitude"
    cutoff_mhz: float = 2000.0  # MHz
    gain_dB_amp: float = 5.0
    gain_dB_phase: float = 5.0
    strength_amp: float =  DetectionConfig.sigma_vac*10**(gain_dB_amp/20) # DetectionConfig.sigma_vac*np.sqrt((10 ** (gain_dB_amp / 10.0) - 1))
    strength_phase: float = DetectionConfig.sigma_vac*10**(gain_dB_phase/20) # DetectionConfig.sigma_vac*np.sqrt((10 ** (gain_dB_phase / 10.0) - 1))
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

def time_axis(duration_ps: float, dt_ps: float) -> Array:
    n = int(np.round(duration_ps / dt_ps))
    return np.arange(n, dtype=np.float64) * dt_ps

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

t = time_axis(duration_ps=4e2, dt_ps=1.0)
psi = integrate_cavity(t, F_t=np.full(t.shape, 0.7 + 0.0j, dtype=np.complex128), cfg=CavityConfig(), integrator="rk4")

plt.plot(t, np.real(psi), label="Re(ψ)")
plt.plot(t, np.imag(psi), label="Im(ψ)")
plt.xlabel("Time (ps)")
plt.ylabel("Intracavity Field")
plt.title("Cavity Dynamics")
plt.legend()
# %%
from Polariton_Microcavity_OHT import HBAR_MEV_PS
print(HBAR_MEV_PS)
# %%
