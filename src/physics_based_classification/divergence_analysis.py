"""
Project SHIELD - Wavelet Divergence Analysis
=============================================
Implements KL divergence and symmetric divergence measures on MODWT
wavelet coefficient distributions for sensor health monitoring.

Methodology (adapted from wavelet divergence-based health monitoring,
ScienceDirect 2025):
  1. Decompose signal into MODWT sub-bands
  2. Build probability distributions of coefficients per sub-band
  3. Establish a reference distribution from a known-good baseline
  4. For each subsequent window, compute divergence from reference
  5. Features with LOW divergence = STABLE (quality vector candidates)
  6. Features with HIGH divergence = SENSITIVE (degradation indicators)

Uses histogram-based distribution estimation (fast, edge-compatible)
with Laplace smoothing to prevent zero-bin KL blow-up.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .wavelet_analyses import modwt_fast, modwt


# =============================================================================
# Distribution Estimation
# =============================================================================

def estimate_distribution(
    coeffs: np.ndarray,
    n_bins: int = 50,
    method: str = 'histogram',
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a probability distribution from wavelet coefficients.

    Parameters
    ----------
    coeffs : 1D array of wavelet coefficients
    n_bins : number of histogram bins
    method : 'histogram' (fast, edge-compatible) or 'kde' (smoother)

    Returns
    -------
    (bin_centers, probabilities) : each 1D array of length n_bins
        probabilities sum to ~1.0
    """
    if method == 'histogram':
        counts, bin_edges = np.histogram(coeffs, bins=n_bins, density=False)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        # Laplace smoothing to avoid zero bins
        probs = (counts + 1.0) / (np.sum(counts) + n_bins)
        return bin_centers, probs

    elif method == 'kde':
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(coeffs, bw_method='silverman')
        x_min, x_max = coeffs.min(), coeffs.max()
        margin = 0.1 * (x_max - x_min + 1e-12)
        bin_centers = np.linspace(x_min - margin, x_max + margin, n_bins)
        probs = kde(bin_centers)
        probs = probs / (np.sum(probs) + 1e-12)
        return bin_centers, probs

    else:
        raise ValueError(f"Unknown method: {method}")


def align_distributions(
    p_centers: np.ndarray, p_probs: np.ndarray,
    q_centers: np.ndarray, q_probs: np.ndarray,
    n_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align two distributions onto a common bin grid via interpolation.

    Necessary when reference and current windows have different value ranges.
    Returns (p_aligned, q_aligned) on the same grid, both summing to ~1.
    """
    all_min = min(p_centers.min(), q_centers.min())
    all_max = max(p_centers.max(), q_centers.max())
    common_grid = np.linspace(all_min, all_max, n_bins)

    p_interp = np.interp(common_grid, p_centers, p_probs, left=0, right=0)
    q_interp = np.interp(common_grid, q_centers, q_probs, left=0, right=0)

    # Re-normalize + smoothing
    eps = 1e-10
    p_interp = p_interp + eps
    q_interp = q_interp + eps
    p_interp = p_interp / np.sum(p_interp)
    q_interp = q_interp / np.sum(q_interp)

    return p_interp, q_interp


# =============================================================================
# Divergence Measures
# =============================================================================

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D_KL(P || Q). Not symmetric."""
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def symmetric_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetric KL divergence: (D_KL(P||Q) + D_KL(Q||P)) / 2.

    Also known as Jeffrey's divergence / 2. More robust than
    one-directional KL for comparing distributions.
    """
    return (kl_divergence(p, q) + kl_divergence(q, p)) / 2.0


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence: JSD(P, Q).

    Bounded [0, ln(2)], symmetric. The square root is a true metric.
    JSD = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M), where M = (P+Q)/2
    """
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


# =============================================================================
# Reference Distribution Building
# =============================================================================

def build_reference_distributions(
    signal: np.ndarray,
    fs: float,
    level: int = 5,
    n_bins: int = 50,
    method: str = 'histogram',
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build reference (baseline) distributions from a known-good signal.

    Parameters
    ----------
    signal : 1D array, the "healthy" / baseline signal
    fs : sampling frequency
    level : MODWT decomposition levels
    n_bins : bins for distribution estimation
    method : 'histogram' or 'kde'

    Returns
    -------
    dict mapping sub-band name ('D1'..'D{level}', 'A{level}') to
    (bin_centers, probabilities) tuples.
    """
    values = np.asarray(signal, dtype=np.float64).flatten()

    if values.size > 256:
        decomp = modwt_fast(values, level=level)
    else:
        decomp = modwt(values, level=level)

    n_levels = decomp['levels']
    ref_dists = {}

    for j in range(1, n_levels + 1):
        centers, probs = estimate_distribution(decomp[f'D{j}'], n_bins, method)
        ref_dists[f'D{j}'] = (centers, probs)

    centers, probs = estimate_distribution(decomp[f'A{n_levels}'], n_bins, method)
    ref_dists[f'A{n_levels}'] = (centers, probs)

    return ref_dists


# =============================================================================
# Per-Window Divergence Computation
# =============================================================================

def compute_window_divergence(
    window: np.ndarray,
    ref_dists: Dict[str, Tuple[np.ndarray, np.ndarray]],
    level: int = 5,
    n_bins: int = 50,
    method: str = 'histogram',
) -> Dict[str, float]:
    """Compute divergence between a window's wavelet distributions and reference.

    Parameters
    ----------
    window : 1D signal window
    ref_dists : reference distributions from build_reference_distributions()
    level : MODWT levels
    n_bins : bins for distribution estimation
    method : distribution estimation method

    Returns
    -------
    dict with keys:
        - div_d{j}_kl : symmetric KL divergence for detail level j
        - div_d{j}_jsd : Jensen-Shannon divergence for detail level j
        - div_a_kl / div_a_jsd : same for approximation
        - div_mean_kl, div_max_kl, div_mean_jsd : summary stats
    """
    values = np.asarray(window, dtype=np.float64).flatten()

    if values.size < 8:
        return {}

    try:
        if values.size > 256:
            decomp = modwt_fast(values, level=level)
        else:
            decomp = modwt(values, level=level)
    except Exception:
        return {}

    n_levels = decomp['levels']
    features = {}
    kl_values = []
    jsd_values = []

    for j in range(1, n_levels + 1):
        band_key = f'D{j}'
        if band_key not in ref_dists:
            continue

        ref_centers, ref_probs = ref_dists[band_key]
        cur_centers, cur_probs = estimate_distribution(
            decomp[band_key], n_bins, method
        )

        p_aligned, q_aligned = align_distributions(
            ref_centers, ref_probs, cur_centers, cur_probs, n_bins
        )

        kl_val = symmetric_kl_divergence(p_aligned, q_aligned)
        jsd_val = jensen_shannon_divergence(p_aligned, q_aligned)

        features[f'div_d{j}_kl'] = kl_val
        features[f'div_d{j}_jsd'] = jsd_val
        kl_values.append(kl_val)
        jsd_values.append(jsd_val)

    # Approximation band
    a_key = f'A{n_levels}'
    if a_key in ref_dists:
        ref_centers, ref_probs = ref_dists[a_key]
        cur_centers, cur_probs = estimate_distribution(
            decomp[a_key], n_bins, method
        )
        p_aligned, q_aligned = align_distributions(
            ref_centers, ref_probs, cur_centers, cur_probs, n_bins
        )

        kl_val = symmetric_kl_divergence(p_aligned, q_aligned)
        jsd_val = jensen_shannon_divergence(p_aligned, q_aligned)

        features['div_a_kl'] = kl_val
        features['div_a_jsd'] = jsd_val
        kl_values.append(kl_val)
        jsd_values.append(jsd_val)

    # Summary statistics
    if kl_values:
        features['div_mean_kl'] = float(np.mean(kl_values))
        features['div_max_kl'] = float(np.max(kl_values))
        features['div_mean_jsd'] = float(np.mean(jsd_values))

    return features


# =============================================================================
# Cross-Sensor / Cross-Dataset Divergence
# =============================================================================

def compute_cross_sensor_divergence(
    ref_dists_a: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ref_dists_b: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 50,
) -> Dict[str, float]:
    """Compute divergence between two sensors' reference distributions.

    Useful for measuring how different two sensors' baseline wavelet
    characteristics are, and which sub-bands are most/least similar.
    """
    features = {}
    common_bands = set(ref_dists_a.keys()) & set(ref_dists_b.keys())

    kl_values = []
    for band in sorted(common_bands):
        ca, pa = ref_dists_a[band]
        cb, pb = ref_dists_b[band]
        p_aligned, q_aligned = align_distributions(ca, pa, cb, pb, n_bins)

        kl_val = symmetric_kl_divergence(p_aligned, q_aligned)
        band_label = band.lower()
        features[f'xdiv_{band_label}_kl'] = kl_val
        kl_values.append(kl_val)

    if kl_values:
        features['xdiv_mean_kl'] = float(np.mean(kl_values))

    return features
