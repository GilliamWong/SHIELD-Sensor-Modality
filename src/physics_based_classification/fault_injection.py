"""
Project SHIELD - Synthetic Fault Injection
==========================================
Reusable synthetic fault injection utilities for validating health-monitoring
pipelines when real degraded sensor data is limited.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


FAULT_TYPES = (
    'noise_increase',
    'bias_drift',
    'spike_injection',
    'bandwidth_loss',
    'saturation',
    'gain_change',
)


def inject_synthetic_fault(
    signal: np.ndarray,
    fault_type: str,
    severity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject a synthetic degradation mode into a 1D or 2D signal.

    Parameters
    ----------
    signal : ndarray
        Shape (n_samples,) or (n_samples, n_channels).
    fault_type : str
        One of FAULT_TYPES.
    severity : float
        Fault intensity. Typical range for validation sweeps is [0.0, 2.0].
    rng : numpy.random.Generator, optional
        Random source for reproducible injections.
    """
    if rng is None:
        rng = np.random.default_rng()

    values = np.asarray(signal, dtype=np.float64).copy()
    original_shape = values.shape

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    elif values.ndim != 2:
        raise ValueError("signal must be 1D or 2D")

    n_samples, n_channels = values.shape
    if n_samples < 4:
        return values.reshape(original_shape)

    sev = max(0.0, float(severity))
    t = np.linspace(0.0, max(sev, 1e-12), n_samples)

    for ch in range(n_channels):
        sig = values[:, ch]
        sig_std = float(np.std(sig) + 1e-8)
        sig_mean = float(np.mean(sig))

        if fault_type == 'noise_increase':
            noise = rng.normal(0.0, 1.0, n_samples)
            envelope = 0.5 + 2.5 * (t / max(sev, 1e-12)) ** 1.5 if sev > 0 else np.zeros(n_samples)
            sig = sig + noise * envelope * sig_std * sev

        elif fault_type == 'bias_drift':
            direction = rng.choice([-1.0, 1.0])
            drift = direction * sig_std * sev * (t / max(sev, 1e-12)) ** 1.5 * 3.0 if sev > 0 else 0.0
            sig = sig + drift

        elif fault_type == 'spike_injection':
            base_prob = 0.02 * sev
            spike_prob = base_prob + (t / max(sev, 1e-12)) * 0.08 * sev if sev > 0 else np.zeros(n_samples)
            spikes = rng.random(n_samples) < spike_prob
            if np.any(spikes):
                signs = rng.choice([-1.0, 1.0], size=int(np.sum(spikes)))
                mags = rng.uniform(3.0, 6.0, size=int(np.sum(spikes))) * sig_std * max(sev, 1e-12)
                sig = sig.copy()
                sig[spikes] += signs * mags

        elif fault_type == 'bandwidth_loss':
            filter_width = int(3 + 20 * sev)
            smoothed = uniform_filter1d(sig, size=max(1, filter_width))
            new_std = float(np.std(smoothed) + 1e-8)
            sig = (smoothed - np.mean(smoothed)) * (sig_std / new_std) * 0.6 + sig_mean

        elif fault_type == 'saturation':
            abs_scale = max(float(np.max(np.abs(sig))), sig_std, 1e-8)
            clip_ratio = float(np.clip(0.95 - 0.30 * sev, 0.12, 0.95))
            threshold = clip_ratio * abs_scale
            sig = np.clip(sig, -threshold, threshold)
            if sev > 0:
                quant_step = max(abs_scale * (0.01 + 0.02 * sev), 1e-8)
                sig = np.round(sig / quant_step) * quant_step

        elif fault_type == 'gain_change':
            if rng.random() < 0.5:
                gain = max(0.1, 1.0 - 0.45 * sev)
            else:
                gain = 1.0 + 1.2 * sev
            sig = sig * gain

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

        values[:, ch] = sig

    return values.reshape(original_shape)
