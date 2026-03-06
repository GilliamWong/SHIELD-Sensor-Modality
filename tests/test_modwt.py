"""
MODWT correctness tests for Project SHIELD.

Tests energy preservation, implementation agreement, noise type
discrimination, and degradation detection sensitivity.
"""

import numpy as np
import pytest

from physics_based_classification.wavelet_analyses import (
    modwt, modwt_fast, extract_wavelet_features,
    inject_degradation,
)


# =============================================================================
# Synthetic data generators
# =============================================================================

def generate_white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(0.0, 1.0, n)


def generate_pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, n)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    scaling = np.ones_like(freqs)
    with np.errstate(divide='ignore'):
        scaling[1:] = 1.0 / np.sqrt(freqs[1:])
    scaling[0] = 0
    X_pink = X * scaling
    signal = np.fft.irfft(X_pink, n=n)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    return signal


def generate_brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.normal(0.0, 1.0, n)
    signal = np.cumsum(white)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    return signal


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.parametrize("name,gen", [
    ("White", generate_white_noise),
    ("Pink", generate_pink_noise),
    ("Brown", generate_brown_noise),
])
def test_energy_preservation(name, gen):
    """MODWT total energy should equal signal energy (Parseval's theorem)."""
    rng = np.random.default_rng(42)
    signal = gen(1024, rng)
    signal_energy = np.sum(signal ** 2)

    decomp = modwt_fast(signal, level=5)
    n_levels = decomp['levels']

    wavelet_energy = 0.0
    for j in range(1, n_levels + 1):
        wavelet_energy += np.sum(decomp[f'D{j}'] ** 2)
    wavelet_energy += np.sum(decomp[f'A{n_levels}'] ** 2)

    ratio = wavelet_energy / signal_energy
    assert abs(ratio - 1.0) < 0.05, (
        f"{name} noise: ratio={ratio:.4f}, expected ~1.0"
    )


def test_modwt_vs_modwt_fast():
    """Direct and FFT MODWT implementations should agree to machine precision."""
    rng = np.random.default_rng(42)
    signal = generate_white_noise(256, rng)

    decomp_direct = modwt(signal, level=4)
    decomp_fft = modwt_fast(signal, level=4)

    for key in ['D1', 'D2', 'D3', 'D4', 'A4']:
        diff = np.max(np.abs(decomp_direct[key] - decomp_fft[key]))
        assert diff < 1e-10, f"{key}: max_diff={diff:.2e}"


@pytest.mark.parametrize("name,gen,expected_hf_lf", [
    ("White", generate_white_noise, "high"),
    ("Brown", generate_brown_noise, "low"),
])
def test_noise_type_discrimination(name, gen, expected_hf_lf):
    """Different noise types should produce distinguishable HF/LF energy ratios."""
    rng = np.random.default_rng(42)
    signal = gen(4096, rng)
    features = extract_wavelet_features(signal, fs=100.0, level=6)
    ratio = features['modwt_energy_ratio_hf_lf']
    if expected_hf_lf == "high":
        assert ratio > 0.5, f"{name}: HF/LF ratio={ratio:.3f}, expected > 0.5"
    else:
        assert ratio < 0.5, f"{name}: HF/LF ratio={ratio:.3f}, expected < 0.5"


@pytest.mark.parametrize("mode,key_feature,direction", [
    ("noise_increase", "modwt_d1_energy", "increase"),
    ("bias_drift", "modwt_a_energy", "increase"),
    ("bandwidth_loss", "modwt_energy_ratio_hf_lf", "decrease"),
])
def test_degradation_sensitivity(mode, key_feature, direction):
    """Wavelet features should respond to injected degradation."""
    rng = np.random.default_rng(42)
    clean = generate_white_noise(2048, rng)

    val_start = extract_wavelet_features(
        inject_degradation(clean, mode, 0.0), fs=100.0, level=5
    ).get(key_feature, float('nan'))

    val_end = extract_wavelet_features(
        inject_degradation(clean, mode, 1.0), fs=100.0, level=5
    ).get(key_feature, float('nan'))

    if direction == "increase":
        assert val_end > val_start, (
            f"{mode}: {key_feature} at sev=1.0 ({val_end:.4f}) "
            f"should be > sev=0.0 ({val_start:.4f})"
        )
    else:
        assert val_end < val_start, (
            f"{mode}: {key_feature} at sev=1.0 ({val_end:.4f}) "
            f"should be < sev=0.0 ({val_start:.4f})"
        )
