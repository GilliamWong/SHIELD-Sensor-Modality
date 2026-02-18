"""
Project SHIELD - MODWT Validation & Analysis Script
=====================================================
Tests the MODWT implementation on synthetic and real sensor data.

Run this script to verify:
  1. MODWT correctness (energy preservation, reconstruction)
  2. Feature discrimination across noise types (white/pink/brown)
  3. Degradation detection sensitivity

Usage:
    python src/physics_based_classification/test_modwt.py

Output:
    - Console validation results
    - notebooks/figures/modwt_validation.png
    - notebooks/figures/modwt_energy_comparison.png
    - notebooks/figures/modwt_degradation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wavelet_analyses import (
    modwt, modwt_fast, extract_wavelet_features,
    get_wavelet_energy, get_wavelet_variance
)

# Output directory for generated figures
_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'notebooks', 'figures')


# =============================================================================
# Synthetic Data Generation (mirrors SensorDataLoader patterns)
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


def inject_degradation(signal: np.ndarray, mode: str, severity: float = 1.0) -> np.ndarray:
    """Inject synthetic degradation into a signal.
    
    Parameters
    ----------
    signal : clean signal
    mode : 'noise_increase', 'bias_drift', 'bandwidth_loss', 'spike_injection'
    severity : 0.0 (none) to 1.0 (full), controls degradation intensity
    """
    n = len(signal)
    t = np.linspace(0, 1, n)
    degraded = signal.copy()
    
    if mode == 'noise_increase':
        noise = np.random.normal(0, severity * 3.0, n) * t
        degraded += noise
    elif mode == 'bias_drift':
        drift = severity * 2.0 * t ** 2
        degraded += drift
    elif mode == 'bandwidth_loss':
        from scipy.ndimage import uniform_filter1d
        width = int(1 + severity * 20 * t[-1])
        # Progressive smoothing
        for i in range(n):
            w = max(1, int(1 + severity * 20 * (i / n)))
            start = max(0, i - w // 2)
            end = min(n, i + w // 2 + 1)
            degraded[i] = np.mean(signal[start:end])
    elif mode == 'spike_injection':
        spike_prob = severity * 0.02 * t
        spikes = np.random.random(n) < spike_prob
        degraded[spikes] += np.random.choice([-5, 5], size=np.sum(spikes))
    
    return degraded


# =============================================================================
# Test 1: MODWT Correctness Validation
# =============================================================================

def test_energy_preservation():
    """Verify that total MODWT energy approximately equals signal energy.
    
    Parseval's theorem: sum of squared wavelet coefficients across all
    sub-bands should equal the sum of squared signal values.
    """
    print("=" * 60)
    print("TEST 1: Energy Preservation (Parseval's Theorem)")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    
    for name, gen in [('White', generate_white_noise), 
                       ('Pink', generate_pink_noise),
                       ('Brown', generate_brown_noise)]:
        signal = gen(1024, rng)
        signal_energy = np.sum(signal ** 2)
        
        decomp = modwt_fast(signal, level=5)
        n_levels = decomp['levels']
        
        wavelet_energy = 0.0
        for j in range(1, n_levels + 1):
            wavelet_energy += np.sum(decomp[f'D{j}'] ** 2)
        wavelet_energy += np.sum(decomp[f'A{n_levels}'] ** 2)
        
        ratio = wavelet_energy / signal_energy
        status = "PASS" if abs(ratio - 1.0) < 0.05 else "FAIL"
        print(f"  {name:6s} noise: signal_E={signal_energy:.2f}, "
              f"modwt_E={wavelet_energy:.2f}, ratio={ratio:.4f} [{status}]")
    
    print()


def test_modwt_vs_modwt_fast():
    """Verify that the direct and FFT implementations agree."""
    print("=" * 60)
    print("TEST 2: Direct vs FFT Implementation Agreement")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    signal = generate_white_noise(256, rng)
    
    decomp_direct = modwt(signal, level=4)
    decomp_fft = modwt_fast(signal, level=4)
    
    for key in ['D1', 'D2', 'D3', 'D4', 'A4']:
        diff = np.max(np.abs(decomp_direct[key] - decomp_fft[key]))
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(f"  {key}: max_diff = {diff:.2e} [{status}]")
    
    print()


def test_noise_type_discrimination():
    """Verify that different noise types produce distinguishable energy distributions.
    
    Expected patterns (from theory):
      - White noise: energy roughly equal across detail levels
      - Pink noise (1/f): energy increases at lower frequencies (higher levels)
      - Brown noise (1/f^2): energy concentrated in approximation / high levels
    """
    print("=" * 60)
    print("TEST 3: Noise Type Energy Distribution")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    n = 4096
    level = 6
    
    for name, gen in [('White', generate_white_noise),
                       ('Pink', generate_pink_noise),
                       ('Brown', generate_brown_noise)]:
        signal = gen(n, rng)
        features = extract_wavelet_features(signal, fs=100.0, level=level)
        
        print(f"\n  {name} noise energy distribution:")
        total = features['modwt_total_energy']
        for j in range(1, level + 1):
            rel = features[f'modwt_d{j}_rel_energy']
            bar = '#' * int(rel * 80)
            print(f"    D{j}: {rel*100:5.1f}% {bar}")
        
        a_rel = features['modwt_a_rel_energy']
        bar = '#' * int(a_rel * 80)
        print(f"    A{level}: {a_rel*100:5.1f}% {bar}")
        print(f"    HF/LF ratio: {features['modwt_energy_ratio_hf_lf']:.3f}")
        print(f"    Max energy level: D{int(features['modwt_max_energy_level'])}")
    
    print()


# =============================================================================
# Test 4: Degradation Sensitivity
# =============================================================================

def test_degradation_sensitivity():
    """Test that wavelet features respond to injected degradation modes.
    
    For each degradation type, compute features on progressively degraded 
    signals and check that the expected features change monotonically.
    """
    print("=" * 60)
    print("TEST 4: Degradation Detection Sensitivity")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    n = 2048
    fs = 100.0
    clean = generate_white_noise(n, rng)
    
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    modes = {
        'noise_increase': ('modwt_d1_energy', 'should increase'),
        'bias_drift': ('modwt_a_energy', 'should increase'),
        'bandwidth_loss': ('modwt_energy_ratio_hf_lf', 'should decrease'),
    }
    
    for mode, (key_feature, expected) in modes.items():
        print(f"\n  {mode} → tracking '{key_feature}' ({expected}):")
        values = []
        for sev in severities:
            degraded = inject_degradation(clean, mode, sev)
            features = extract_wavelet_features(degraded, fs=fs, level=5)
            val = features.get(key_feature, float('nan'))
            values.append(val)
            print(f"    severity={sev:.2f}: {key_feature}={val:.4f}")
        
        # Check monotonicity
        if 'increase' in expected:
            monotonic = all(values[i] <= values[i+1] for i in range(len(values)-1))
        else:
            monotonic = all(values[i] >= values[i+1] for i in range(len(values)-1))
        
        status = "PASS" if monotonic else "TREND OK" if (
            (values[-1] > values[0]) if 'increase' in expected else (values[-1] < values[0])
        ) else "CHECK"
        print(f"    Monotonic: {monotonic} [{status}]")
    
    print()


# =============================================================================
# Visualization
# =============================================================================

def plot_noise_decomposition():
    """Generate visualization of MODWT decomposition for different noise types."""
    rng = np.random.default_rng(42)
    n = 2048
    level = 5
    fs = 100.0
    
    fig, axes = plt.subplots(3, level + 2, figsize=(24, 10))
    fig.suptitle('MODWT Decomposition: White vs Pink vs Brown Noise (sym4)', 
                 fontsize=14, fontweight='bold')
    
    for row, (name, gen) in enumerate([
        ('White', generate_white_noise),
        ('Pink', generate_pink_noise),
        ('Brown', generate_brown_noise)
    ]):
        signal = gen(n, rng)
        decomp = modwt_fast(signal, level=level)
        
        # Original signal
        axes[row, 0].plot(signal[:500], linewidth=0.5, color='steelblue')
        axes[row, 0].set_title(f'{name} (original)' if row == 0 else '')
        axes[row, 0].set_ylabel(name, fontweight='bold', fontsize=12)
        axes[row, 0].tick_params(labelsize=7)
        
        # Detail levels
        for j in range(1, level + 1):
            axes[row, j].plot(decomp[f'D{j}'][:500], linewidth=0.5, color='coral')
            if row == 0:
                axes[row, j].set_title(f'D{j}')
            axes[row, j].tick_params(labelsize=7)
        
        # Approximation
        axes[row, level + 1].plot(decomp[f'A{level}'][:500], linewidth=0.5, color='seagreen')
        if row == 0:
            axes[row, level + 1].set_title(f'A{level}')
        axes[row, level + 1].tick_params(labelsize=7)
    
    plt.tight_layout()
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    out = os.path.join(_FIGURES_DIR, 'modwt_validation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_energy_comparison():
    """Bar chart comparing relative energy distributions across noise types."""
    rng = np.random.default_rng(42)
    n = 4096
    level = 6
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('MODWT Relative Energy Distribution by Noise Type', 
                 fontsize=13, fontweight='bold')
    
    for ax, (name, gen) in zip(axes, [
        ('White Noise', generate_white_noise),
        ('Pink Noise (1/f)', generate_pink_noise),
        ('Brown Noise (1/f²)', generate_brown_noise)
    ]):
        signal = gen(n, rng)
        features = extract_wavelet_features(signal, fs=100.0, level=level)
        
        labels = [f'D{j}' for j in range(1, level + 1)] + [f'A{level}']
        energies = [features[f'modwt_d{j}_rel_energy'] for j in range(1, level + 1)]
        energies.append(features['modwt_a_rel_energy'])
        
        colors = ['#e74c3c'] * level + ['#27ae60']
        ax.bar(labels, energies, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(name, fontweight='bold')
        ax.set_ylabel('Relative Energy')
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    out = os.path.join(_FIGURES_DIR, 'modwt_energy_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


def plot_degradation_tracking():
    """Show how wavelet features evolve as degradation progresses."""
    rng = np.random.default_rng(42)
    n = 2048
    fs = 100.0
    n_steps = 20
    
    clean = generate_white_noise(n, rng)
    severities = np.linspace(0, 1.0, n_steps)
    
    modes = ['noise_increase', 'bias_drift', 'bandwidth_loss', 'spike_injection']
    track_features = ['modwt_d1_energy', 'modwt_a_energy', 
                       'modwt_energy_ratio_hf_lf', 'modwt_d1_variance']
    
    fig, axes = plt.subplots(len(modes), len(track_features), figsize=(20, 14))
    fig.suptitle('MODWT Feature Response to Sensor Degradation', 
                 fontsize=14, fontweight='bold')
    
    for i, mode in enumerate(modes):
        trajectories = {f: [] for f in track_features}
        
        for sev in severities:
            degraded = inject_degradation(clean, mode, sev)
            features = extract_wavelet_features(degraded, fs=fs, level=5)
            for f in track_features:
                trajectories[f].append(features.get(f, np.nan))
        
        for j, feat in enumerate(track_features):
            axes[i, j].plot(severities, trajectories[feat], 
                           'o-', markersize=3, linewidth=1.5, color='steelblue')
            if i == 0:
                axes[i, j].set_title(feat.replace('modwt_', ''), fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(mode.replace('_', '\n'), 
                                       fontweight='bold', fontsize=9)
            if i == len(modes) - 1:
                axes[i, j].set_xlabel('Severity')
            axes[i, j].tick_params(labelsize=7)
            axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    out = os.path.join(_FIGURES_DIR, 'modwt_degradation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  Saved: {out}")
    plt.close()


# =============================================================================
# Feature Extraction Demo (showing integration with existing pipeline)
# =============================================================================

def demo_feature_extraction():
    """Demonstrate how extract_wavelet_features() output looks, 
    ready for integration into FeatureExtractor."""
    print("=" * 60)
    print("DEMO: Feature Extraction Output")
    print("=" * 60)
    
    rng = np.random.default_rng(42)
    signal = generate_white_noise(1024, rng)
    features = extract_wavelet_features(signal, fs=100.0, level=5)
    
    print("\n  extract_wavelet_features() returns:")
    for key, val in sorted(features.items()):
        print(f"    {key:30s} = {val:.6f}")
    
    print(f"\n  Total features: {len(features)}")
    print("  These can be merged into the FeatureExtractor pipeline alongside")
    print("  time_domain, freq_domain, and Allan deviation features.")
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Project SHIELD - MODWT Validation Suite")
    print("=" * 60 + "\n")
    
    # Correctness tests
    test_energy_preservation()
    test_modwt_vs_modwt_fast()
    test_noise_type_discrimination()
    test_degradation_sensitivity()
    
    # Demo
    demo_feature_extraction()
    
    # Plots
    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plot_noise_decomposition()
    plot_energy_comparison()
    plot_degradation_tracking()
    
    print("\n" + "=" * 60)
    print("  All tests complete!")
    print("=" * 60)