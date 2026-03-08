"""
Project SHIELD - Signal Quality Features
=========================================
Time-domain signal quality metrics for sensor health assessment.

These features measure the intrinsic quality of a sensor signal window,
independent of the specific physical quantity being measured. They detect
degradation modes mapped to the fault taxonomy from Kuzin & Borovicka:
  - Noise floor elevation / high noise  (noise_floor, snr)
  - Calibration drift                   (baseline_stability)
  - Stuck-at / data loss                (dropout_rate)
  - Outlier / spike                     (peak_consistency)
  - Nonlinear distortion                (response_linearity)

All features are computed per-window and returned as Dict[str, float]
following the existing SHIELD pipeline convention. All computations are
O(N) — suitable for eventual CMSIS-DSP translation.
"""

import numpy as np
from typing import Dict


def get_noise_floor(signal: np.ndarray, fs: float) -> float:
    """Estimate noise floor as MAD of the first-differenced signal.

    First-differencing removes trend/DC content, isolating high-frequency
    noise. MAD scaled by 1.4826 approximates Gaussian sigma, divided by
    sqrt(2) to correct for the differencing operation.

    Lower values = cleaner signal. Rising values = noise floor elevation.
    """
    diff = np.diff(signal)
    mad = np.median(np.abs(diff - np.median(diff)))
    return float(mad * 1.4826 / np.sqrt(2.0))


def get_baseline_stability(signal: np.ndarray, fs: float,
                           n_segments: int = 4) -> float:
    """Measure baseline stability as std of segment means.

    Splits the window into n_segments equal parts, computes each mean,
    then returns the std. A stable sensor has near-zero variation;
    a drifting sensor shows increasing variation.
    """
    n = len(signal)
    seg_len = n // n_segments
    if seg_len < 2:
        return 0.0
    seg_means = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len
        seg_means.append(np.mean(signal[start:end]))
    return float(np.std(seg_means))


def get_dropout_rate(signal: np.ndarray, fs: float,
                     repeat_thresh: int = 3) -> float:
    """Detect data dropouts as sequences of repeated identical values.

    A healthy sensor produces continuously varying output. Repeated
    identical values indicate sample-and-hold failures, communication
    dropouts, or stuck-at faults.

    Returns the fraction of samples in repeated runs of length >= repeat_thresh.
    """
    n = len(signal)
    if n < repeat_thresh:
        return 0.0

    dropout_count = 0
    run_length = 1

    for i in range(1, n):
        if signal[i] == signal[i - 1]:
            run_length += 1
        else:
            if run_length >= repeat_thresh:
                dropout_count += run_length
            run_length = 1
    if run_length >= repeat_thresh:
        dropout_count += run_length

    return float(dropout_count / n)


def get_peak_consistency(signal: np.ndarray, fs: float) -> float:
    """Measure CV of peak-to-peak amplitudes in overlapping sub-segments.

    Consistent peaks indicate a stable sensor. Erratic peaks suggest
    intermittent faults, spikes, or saturation events.
    Lower CV = more consistent.
    """
    n = len(signal)
    seg_size = max(n // 8, 4)
    step = max(seg_size // 2, 1)

    ptps = []
    for start in range(0, n - seg_size + 1, step):
        seg = signal[start:start + seg_size]
        ptps.append(np.ptp(seg))

    if len(ptps) < 2:
        return 0.0

    ptps = np.array(ptps)
    mean_ptp = np.mean(ptps)
    if mean_ptp < 1e-12:
        return 0.0
    return float(np.std(ptps) / mean_ptp)


def get_snr(signal: np.ndarray, fs: float) -> float:
    """Estimate signal-to-noise ratio in dB.

    Uses signal variance vs noise variance estimated from first-differencing.
    SNR = 10 * log10(signal_variance / noise_variance)

    Positive dB = signal dominates. Negative = noise dominates.
    Falling SNR indicates sensor degradation.
    """
    sig_var = np.var(signal)
    diff = np.diff(signal)
    noise_var = np.var(diff) / 2.0  # Correct for differencing

    if noise_var < 1e-20:
        return 60.0  # Effectively noiseless (cap at 60 dB)
    if sig_var < 1e-20:
        return -60.0  # Effectively no signal

    return float(10.0 * np.log10(sig_var / noise_var))


def get_response_linearity(signal: np.ndarray, fs: float) -> float:
    """Measure how well the signal follows a linear trend (R-squared).

    Values near 1.0 = linear/constant signal. Lower values = oscillatory
    or nonlinear content. A sudden drop in linearity (for a sensor that
    should be linear) indicates distortion.
    """
    n = len(signal)
    if n < 3:
        return 1.0
    t = np.arange(n, dtype=np.float64)
    t_mean = np.mean(t)
    s_mean = np.mean(signal)
    ss_tt = np.sum((t - t_mean) ** 2)
    if ss_tt < 1e-12:
        return 1.0
    slope = np.sum((t - t_mean) * (signal - s_mean)) / ss_tt
    intercept = s_mean - slope * t_mean
    predicted = slope * t + intercept
    ss_res = np.sum((signal - predicted) ** 2)
    ss_tot = np.sum((signal - s_mean) ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


def extract_signal_quality_features(
    signal: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    """Extract all signal quality features from a sensor window.

    Parameters
    ----------
    signal : 1D numpy array
    fs : float, sampling frequency in Hz

    Returns
    -------
    dict with keys: sq_noise_floor, sq_baseline_stability, sq_dropout_rate,
                    sq_peak_consistency, sq_snr, sq_response_linearity
    """
    values = np.asarray(signal, dtype=np.float64).flatten()

    if values.size < 8:
        return {}

    try:
        return {
            'sq_noise_floor': get_noise_floor(values, fs),
            'sq_baseline_stability': get_baseline_stability(values, fs),
            'sq_dropout_rate': get_dropout_rate(values, fs),
            'sq_peak_consistency': get_peak_consistency(values, fs),
            'sq_snr': get_snr(values, fs),
            'sq_response_linearity': get_response_linearity(values, fs),
        }
    except Exception:
        return {}
