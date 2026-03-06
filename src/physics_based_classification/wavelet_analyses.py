"""
Project SHIELD - Wavelet Domain Analyses
=========================================
Implements the Maximal Overlap Discrete Wavelet Transform (MODWT) and
extracts wavelet-domain features for sensor signal characterization.

MODWT advantages over standard DWT:
  - Shift-invariant (critical for real-time sensor monitoring)
  - Works with arbitrary signal lengths (no power-of-2 restriction)
  - Produces N coefficients per level (no downsampling)

Reference: Percival & Walden (2000), "Wavelet Methods for Time Series Analysis"
           Cambridge University Press, Chapter 5.

The MODWT decomposes a signal into detail coefficients D1..Dj (high-to-low
frequency bands) and approximation coefficients Aj (baseline trend).
For sensor degradation:
  - D1 (highest freq): captures noise-like content, noise elevation
  - D2-D5 (mid freq):  captures transients, oscillatory behavior
  - A5 (lowest freq):  captures slow trends like bias drift
"""

import numpy as np
from typing import Dict, Mapping, Optional, Sequence, Tuple

# =============================================================================
# Sym4 (Daubechies least-asymmetric, 4 vanishing moments) filter coefficients
# These are the standard scaling (lowpass) filter coefficients.
# Reference: Percival & Walden (2000), Table 107; also scipy.signal / pywt
# =============================================================================
_SYM4_SCALING = np.array([
    -0.07576571478927333,
    -0.02963552764599851,
     0.49761866763201545,
     0.80373875180591614,
     0.29785779560527736,
    -0.09921954357684722,
    -0.01260396726203783,
     0.03222310060404270,
])

def _build_wavelet_filter(scaling: np.ndarray) -> np.ndarray:
    """Derive the wavelet (highpass) filter from the scaling (lowpass) filter.
    
    Uses the quadrature mirror relationship:
        h[n] = (-1)^n * g[L-1-n]
    where g is the scaling filter and L is the filter length.
    """
    L = len(scaling)
    wavelet = np.zeros(L)
    for n in range(L):
        wavelet[n] = ((-1) ** n) * scaling[L - 1 - n]
    return wavelet


# Pre-compute the base filters
_SYM4_WAVELET = _build_wavelet_filter(_SYM4_SCALING)


# =============================================================================
# Core MODWT Implementation
# =============================================================================

def _max_modwt_level(N: int, L: int) -> int:
    """Compute the maximum safe MODWT decomposition level.

    At level j, the upsampled filter spans (L-1)*2^(j-1) + 1 samples.
    For FFT-based circular convolution, this must be <= N to avoid
    index collisions.
    """
    if N < 2 or L < 2:
        return 0
    return max(1, int(np.floor(np.log2(N / (L - 1)))))


def modwt(signal: np.ndarray, wavelet: str = 'sym4', level: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Compute the Maximal Overlap Discrete Wavelet Transform (MODWT).
    
    Decomposes the input signal into detail coefficients at each level
    (D1, D2, ..., Dj) and approximation coefficients at the final level (Aj).
    
    Parameters
    ----------
    signal : 1D numpy array
        Input signal to decompose.
    wavelet : str, default 'sym4'
        Wavelet family to use. Currently supports 'sym4' (Daubechies 
        least-asymmetric with 4 vanishing moments), which is recommended 
        for sensor signals per Percival & Walden.
    level : int or None
        Number of decomposition levels. If None, uses the maximum level:
        floor(log2(N / (L-1) + 1)) where N is signal length and L is 
        filter length.
    
    Returns
    -------
    dict with keys:
        'D1', 'D2', ..., 'Dj' : detail coefficients at each level
        'Aj' : approximation (smooth) coefficients at final level
        'levels' : int, number of decomposition levels
    
    Notes
    -----
    The MODWT is computed using the pyramid algorithm:
      1. Start with V0 = signal
      2. At each level j:
         - Wj = wavelet_filter * V_{j-1}  (detail coefficients)
         - Vj = scaling_filter * V_{j-1}   (approximation coefficients)
      3. The MODWT filters are the DWT filters rescaled by 1/sqrt(2)
    """
    values = np.asarray(signal, dtype=np.float64).flatten()
    N = values.size
    
    if N < 2:
        return {'levels': 0}
    
    # Select wavelet filters
    if wavelet == 'sym4':
        g = _SYM4_SCALING.copy()  # scaling (lowpass)
        h = _SYM4_WAVELET.copy()  # wavelet (highpass)
    else:
        raise ValueError(f"Wavelet '{wavelet}' not supported. Use 'sym4'.")
    
    L = len(g)
    max_level = _max_modwt_level(N, L)

    # Determine max level if not specified
    if level is None:
        level = max_level
    else:
        level = min(level, max_level)

    # Pyramid algorithm for MODWT
    # MODWT filters at level j are rescaled by 1/sqrt(2) relative to level j-1
    result = {}
    V = values.copy()  # V0 = original signal
    
    for j in range(1, level + 1):
        # MODWT rescaled filters: divide base filters by sqrt(2)
        g_modwt = g / np.sqrt(2.0)
        h_modwt = h / np.sqrt(2.0)
        
        # Upsample factor for this level relative to current V
        # In pyramid algorithm, level is always 1 relative to current V
        # but we need to upsample by 2^(j-1) from the original signal perspective
        # However, in pyramid form, we just use the base filters each time
        # because V already represents the approximation from the previous level.
        
        # For the pyramid MODWT, we use circular convolution with stride 2^(j-1)
        # Actually, the standard pyramid MODWT algorithm:
        #   W_j[t] = sum_{l=0}^{L-1} h_tilde[l] * V_{j-1}[t - 2^(j-1)*l mod N]
        #   V_j[t] = sum_{l=0}^{L-1} g_tilde[l] * V_{j-1}[t - 2^(j-1)*l mod N]
        
        stride = 2 ** (j - 1)
        W_j = np.zeros(N)
        V_j = np.zeros(N)
        
        for t in range(N):
            w_sum = 0.0
            v_sum = 0.0
            for l in range(L):
                idx = (t - stride * l) % N
                w_sum += h_modwt[l] * V[idx]
                v_sum += g_modwt[l] * V[idx]
            W_j[t] = w_sum
            V_j[t] = v_sum
        
        result[f'D{j}'] = W_j
        V = V_j
    
    result[f'A{level}'] = V
    result['levels'] = level
    
    return result


def modwt_fast(signal: np.ndarray, wavelet: str = 'sym4', level: Optional[int] = None) -> Dict[str, np.ndarray]:
    """FFT-accelerated MODWT for longer signals.
    
    Equivalent to modwt() but uses FFT-based circular convolution for
    O(N log N) performance instead of O(N * L * J). Use this for signals
    longer than ~1000 samples.
    
    Parameters
    ----------
    signal : 1D numpy array
    wavelet : str, default 'sym4'
    level : int or None
    
    Returns
    -------
    Same dict structure as modwt()
    """
    values = np.asarray(signal, dtype=np.float64).flatten()
    N = values.size
    
    if N < 2:
        return {'levels': 0}
    
    if wavelet == 'sym4':
        g = _SYM4_SCALING.copy()
        h = _SYM4_WAVELET.copy()
    else:
        raise ValueError(f"Wavelet '{wavelet}' not supported. Use 'sym4'.")
    
    L = len(g)
    max_level = _max_modwt_level(N, L)

    if level is None:
        level = max_level
    else:
        level = min(level, max_level)

    # MODWT base filters (rescaled by 1/sqrt(2))
    g_modwt = g / np.sqrt(2.0)
    h_modwt = h / np.sqrt(2.0)
    
    result = {}
    V = values.copy()
    
    for j in range(1, level + 1):
        stride = 2 ** (j - 1)
        
        # Build the effective upsampled filter for this level
        h_j = np.zeros(N)
        g_j = np.zeros(N)
        for l in range(L):
            idx = (l * stride) % N
            h_j[idx] = h_modwt[l]
            g_j[idx] = g_modwt[l]
        
        # FFT-based circular convolution
        V_fft = np.fft.fft(V)
        W_j = np.real(np.fft.ifft(V_fft * np.fft.fft(h_j)))
        V_new = np.real(np.fft.ifft(V_fft * np.fft.fft(g_j)))
        
        result[f'D{j}'] = W_j
        V = V_new
    
    result[f'A{level}'] = V
    result['levels'] = level
    
    return result


# =============================================================================
# Wavelet Feature Extraction
# =============================================================================

def get_wavelet_energy(coeffs: np.ndarray) -> float:
    """Compute the energy (sum of squared coefficients) of a wavelet sub-band."""
    return float(np.sum(coeffs ** 2))


def get_wavelet_variance(coeffs: np.ndarray) -> float:
    """Compute the variance of wavelet coefficients in a sub-band.
    
    The MODWT wavelet variance at scale j is:
        var_j = Var(W_j)
    For detail coefficients this is equivalent to mean(W_j^2), because the
    wavelet filter has zero DC gain and detail bands are zero-mean up to
    floating-point precision. For the final approximation band, centering is
    required so slowly varying offsets are not mislabeled as variance.
    """
    return float(np.var(coeffs))


def get_wavelet_rms(coeffs: np.ndarray) -> float:
    """Compute the RMS of wavelet coefficients in a sub-band.

    RMS measures the average amplitude of oscillatory content at that
    frequency scale. Equivalent to sqrt(variance) for zero-mean MODWT
    coefficients.
    """
    return float(np.sqrt(np.mean(coeffs ** 2)))


def get_wavelet_kurtosis(coeffs: np.ndarray) -> float:
    """Compute the excess kurtosis of wavelet coefficients in a sub-band.

    High kurtosis = heavy tails = transient/impulse content (spikes).
    Low kurtosis = light tails = smooth/Gaussian content.
    Uses Fisher's definition (excess kurtosis; normal distribution = 0).
    """
    n = len(coeffs)
    if n < 4:
        return 0.0
    mean_c = np.mean(coeffs)
    std_c = np.std(coeffs)
    if std_c < 1e-12:
        return 0.0
    m4 = np.mean((coeffs - mean_c) ** 4)
    return float(m4 / (std_c ** 4) - 3.0)


def get_wavelet_entropy(coeffs: np.ndarray) -> float:
    """Compute the Shannon entropy of the energy distribution within a sub-band.
    
    Measures how "spread out" the energy is across the coefficients.
    Low entropy = energy concentrated (transient/impulse).
    High entropy = energy distributed (noise-like).
    """
    eps = 1e-12
    c2 = coeffs ** 2
    total = np.sum(c2) + eps
    p = c2 / total
    p = np.clip(p, eps, None)
    return float(-np.sum(p * np.log2(p)))


def extract_wavelet_features(
    signal: np.ndarray,
    fs: float,
    level: Optional[int] = None,
    wavelet: str = 'sym4',
    use_fft: bool = True,
) -> Mapping[str, float]:
    """Extract wavelet-domain features from a sensor signal using MODWT.
    
    Computes a feature dictionary from MODWT decomposition, following the 
    same pattern as extract_freq_features() and get_statistical_moments()
    in the existing SHIELD pipeline.
    
    Parameters
    ----------
    signal : 1D numpy array
        Raw sensor signal.
    fs : float
        Sampling frequency in Hz.
    level : int or None
        Number of MODWT decomposition levels. Default None = auto.
    wavelet : str
        Wavelet to use. Default 'sym4'.
    use_fft : bool
        If True, use FFT-accelerated MODWT (recommended for N > 1000).
    
    Returns
    -------
    dict of {feature_name: float} with keys:
        - modwt_d{j}_energy:    energy of detail level j
        - modwt_d{j}_variance:  wavelet variance at scale j
        - modwt_d{j}_entropy:   entropy of detail level j
        - modwt_a_energy:       energy of final approximation
        - modwt_a_variance:     variance of final approximation
        - modwt_a_entropy:      entropy of final approximation
        - modwt_d{j}_rel_energy: relative energy (fraction of total)
        - modwt_total_energy:   total signal energy across all sub-bands
        - modwt_energy_ratio_hf_lf: ratio of high-freq to low-freq energy
          (D1+D2 vs A_level), useful for noise detection
        - modwt_max_energy_level: which detail level has the most energy
        - modwt_levels:         number of decomposition levels used
    
    Notes
    -----
    Feature naming convention matches the existing SHIELD pipeline style.
    These features are designed to complement, not replace, the existing
    PSD and Allan deviation features.
    
    For degradation monitoring:
        - Rising D1/D2 energy → noise elevation
        - Rising A_level energy → bias drift
        - Shift in max_energy_level → changing noise characteristics
        - Decreasing energy_ratio_hf_lf → bandwidth loss / sluggishness
    """
    values = np.asarray(signal, dtype=np.float64).flatten()
    
    if values.size < 8:
        return {}
    
    # Compute MODWT decomposition
    if use_fft and values.size > 256:
        decomp = modwt_fast(values, wavelet=wavelet, level=level)
    else:
        decomp = modwt(values, wavelet=wavelet, level=level)
    
    n_levels = decomp['levels']
    if n_levels == 0:
        return {}
    
    features = {}
    features['modwt_levels'] = float(n_levels)
    
    # Per-level features for detail coefficients
    total_energy = 0.0
    detail_energies = []
    
    for j in range(1, n_levels + 1):
        Dj = decomp[f'D{j}']
        e = get_wavelet_energy(Dj)
        v = get_wavelet_variance(Dj)
        ent = get_wavelet_entropy(Dj)
        
        rms_val = get_wavelet_rms(Dj)
        kurt_val = get_wavelet_kurtosis(Dj)

        features[f'modwt_d{j}_energy'] = e
        features[f'modwt_d{j}_variance'] = v
        features[f'modwt_d{j}_entropy'] = ent
        features[f'modwt_d{j}_rms'] = rms_val
        features[f'modwt_d{j}_kurtosis'] = kurt_val

        total_energy += e
        detail_energies.append(e)
    
    # Approximation coefficients
    A = decomp[f'A{n_levels}']
    a_energy = get_wavelet_energy(A)
    features['modwt_a_energy'] = a_energy
    features['modwt_a_variance'] = get_wavelet_variance(A)
    features['modwt_a_entropy'] = get_wavelet_entropy(A)
    features['modwt_a_rms'] = get_wavelet_rms(A)
    features['modwt_a_kurtosis'] = get_wavelet_kurtosis(A)
    total_energy += a_energy
    
    # Total energy and relative energies
    features['modwt_total_energy'] = total_energy
    eps = 1e-12
    
    for j in range(1, n_levels + 1):
        features[f'modwt_d{j}_rel_energy'] = detail_energies[j - 1] / (total_energy + eps)
    
    features['modwt_a_rel_energy'] = a_energy / (total_energy + eps)
    
    # High-frequency to low-frequency energy ratio
    # D1 + D2 = highest frequency content; A_level = lowest frequency content
    hf_energy = sum(detail_energies[:min(2, n_levels)])
    lf_energy = a_energy + eps
    features['modwt_energy_ratio_hf_lf'] = hf_energy / lf_energy
    
    # Which detail level has the most energy (indicates dominant noise type)
    if detail_energies:
        features['modwt_max_energy_level'] = float(np.argmax(detail_energies) + 1)

    return features


# =============================================================================
# Synthetic Degradation Injection
# =============================================================================

def inject_degradation(signal: np.ndarray, mode: str, severity: float = 1.0) -> np.ndarray:
    """Inject synthetic degradation into a signal.

    Parameters
    ----------
    signal : clean signal (1D array)
    mode : 'noise_increase', 'bias_drift', 'bandwidth_loss', 'spike_injection'
    severity : 0.0 (none) to 1.0 (full), controls degradation intensity

    Returns
    -------
    Degraded copy of the signal.
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
        for i in range(n):
            w = max(1, int(1 + severity * 20 * (i / n)))
            start = max(0, i - w // 2)
            end = min(n, i + w // 2 + 1)
            degraded[i] = np.mean(signal[start:end])
    elif mode == 'spike_injection':
        spike_prob = severity * 0.02 * t
        spikes = np.random.random(n) < spike_prob
        degraded[spikes] += np.random.choice([-5, 5], size=np.sum(spikes))
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

    return degraded
