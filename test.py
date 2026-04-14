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
print(data["F_t"]) 
# %%
