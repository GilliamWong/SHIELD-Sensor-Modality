import numpy as np
from scipy.signal import welch
from scipy.stats import entropy

#computes the Power Spectral Density (PSD) using Welch's method.
def get_psd_welch(signal: np.ndarray, fs: float) -> tuple:
    freqs, power = welch(signal, fs = fs, nperseg = 256)
    return (freqs, power)

#calculates the slope of the log log PSD.
def get_spectral_slope(freqs: np.ndarray, power: np.ndarray) -> float:
    mask = freqs > 0
    freqs = freqs[mask]
    power = power[mask]
    
    log_freqs = np.log10(freqs)
    log_power = np.log10(power)

    valid = np.isfinite(log_freqs) & np.isfinite(log_power)
    log_freqs = log_freqs[valid]
    log_power = log_power[valid]
    
    slope, _ = np.polyfit(log_freqs, log_power, 1)
    return slope
    
#calculates the Shannon Entropy of the spectrum.
def get_spectral_entropy(power: np.ndarray) -> float:
    power = power / np.sum(power)
    eps = 1e-12
    power = np.clip(power, eps, 1.0)
    return entropy(power)