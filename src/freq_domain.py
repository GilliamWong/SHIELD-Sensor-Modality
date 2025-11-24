import numpy as np
from scipy.signal import welch
from scipy.stats import entropy
from typing import Iterable, Mapping, Sequence, Tuple

#computes the Power Spectral Density (PSD) using Welch's method.
def get_psd_welch(signal: np.ndarray, fs: float, nperseg: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(signal, dtype=np.float64).flatten()
    if values.size < 2:
        return np.array([]), np.array([])
    nperseg = min(nperseg, values.shape[0])
    freqs, power = welch(values, fs=fs, nperseg=nperseg, scaling="density")
    return freqs, power

#calculates the slope of the log log PSD.
def get_spectral_slope(freqs: np.ndarray, power: np.ndarray) -> float:
    mask = (freqs > 0) & (power > 0)
    freqs = freqs[mask]
    power = power[mask]

    if freqs.size < 2:
        return np.nan
    
    log_freqs = np.log10(freqs)
    log_power = np.log10(power)

    valid = np.isfinite(log_freqs) & np.isfinite(log_power)
    if not np.any(valid):
        return np.nan

    slope, _ = np.polyfit(log_freqs[valid], log_power[valid], 1)
    return slope
    
#calculates the Shannon Entropy of the spectrum.
def get_spectral_entropy(power: np.ndarray) -> float:
    eps = 1e-12
    power = np.clip(power, eps, None)
    power = power / np.sum(power)
    return float(entropy(power))

def get_spectral_flatness(power: np.ndarray) -> float:
    eps = 1e-12
    power = np.clip(power, eps, None)
    geometric_mean = np.exp(np.mean(np.log(power)))
    arithmetic_mean = np.mean(power)
    if arithmetic_mean == 0:
        return np.nan
    return float(geometric_mean / arithmetic_mean)

def get_spectral_centroid(freqs: np.ndarray, power: np.ndarray) -> float:
    eps = 1e-12
    total_power = np.sum(power) + eps
    return float(np.sum(freqs * power) / total_power)

def get_band_energy(freqs: np.ndarray, power: np.ndarray, band: Sequence[float]) -> float:
    low, high = band
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(power[mask]))

def extract_freq_features(signal: np.ndarray, fs: float, bands: Iterable[Sequence[float]] = None) -> Mapping[str, float]:
    freqs, power = get_psd_welch(signal, fs)
    if freqs.size == 0:
        return {}

    features = {
        'psd_slope': get_spectral_slope(freqs, power),
        'spectral_entropy': get_spectral_entropy(power),
        'spectral_flatness': get_spectral_flatness(power),
        'spectral_centroid': get_spectral_centroid(freqs, power),
    }

    if bands is not None:
        for band in bands:
            low, high = band
            key = f"band_energy_{low:.1f}_{high:.1f}"
            features[key] = get_band_energy(freqs, power, band)

    return features
