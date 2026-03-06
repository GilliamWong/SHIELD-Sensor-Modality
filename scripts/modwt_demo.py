#!/usr/bin/env python
"""
Project SHIELD - MODWT Visualization & Demo Script
===================================================
Generates MODWT decomposition plots and demonstrates feature extraction.

Usage:
    python scripts/modwt_demo.py

Output:
    - notebooks/figures/modwt_validation.png
    - notebooks/figures/modwt_energy_comparison.png
    - notebooks/figures/modwt_degradation.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from physics_based_classification.wavelet_analyses import (
    modwt_fast, extract_wavelet_features, inject_degradation,
)

# Output directory for generated figures
_FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'notebooks', 'figures')


# =============================================================================
# Synthetic data generators
# =============================================================================

def generate_white_noise(n, rng):
    return rng.normal(0.0, 1.0, n)


def generate_pink_noise(n, rng):
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


def generate_brown_noise(n, rng):
    white = rng.normal(0.0, 1.0, n)
    signal = np.cumsum(white)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-12)
    return signal


# =============================================================================
# Visualizations
# =============================================================================

def plot_noise_decomposition():
    """Generate visualization of MODWT decomposition for different noise types."""
    rng = np.random.default_rng(42)
    n = 2048
    level = 5

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

        axes[row, 0].plot(signal[:500], linewidth=0.5, color='steelblue')
        axes[row, 0].set_title(f'{name} (original)' if row == 0 else '')
        axes[row, 0].set_ylabel(name, fontweight='bold', fontsize=12)
        axes[row, 0].tick_params(labelsize=7)

        for j in range(1, level + 1):
            axes[row, j].plot(decomp[f'D{j}'][:500], linewidth=0.5, color='coral')
            if row == 0:
                axes[row, j].set_title(f'D{j}')
            axes[row, j].tick_params(labelsize=7)

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


def demo_feature_extraction():
    """Demonstrate extract_wavelet_features() output."""
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
    print()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Project SHIELD - MODWT Demo & Visualization")
    print("=" * 60 + "\n")

    demo_feature_extraction()

    print("=" * 60)
    print("Generating Visualizations")
    print("=" * 60)
    plot_noise_decomposition()
    plot_energy_comparison()
    plot_degradation_tracking()

    print("\nDone!")
