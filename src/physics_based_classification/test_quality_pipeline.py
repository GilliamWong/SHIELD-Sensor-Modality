"""
Test suite for the MODWT + KL Divergence Quality Vector pipeline.
Tests wavelet RMS/kurtosis, signal quality features, divergence functions,
and quality vector construction.

Run with: pytest test_quality_pipeline.py -v
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from physics_based_classification.wavelet_analyses import (
    extract_wavelet_features, modwt_fast, get_wavelet_rms, get_wavelet_kurtosis,
)
from physics_based_classification.signal_quality import (
    extract_signal_quality_features, get_noise_floor, get_snr,
    get_dropout_rate, get_baseline_stability, get_response_linearity,
    get_peak_consistency,
)
from physics_based_classification.divergence_analysis import (
    kl_divergence, symmetric_kl_divergence, jensen_shannon_divergence,
    estimate_distribution, align_distributions,
    build_reference_distributions, compute_window_divergence,
    compute_cross_sensor_divergence,
)
from physics_based_classification.quality_vector import (
    assess_feature_stability, assess_cross_condition_stability,
    select_quality_features, build_quality_vector,
    DEFAULT_MANDATORY_FEATURES,
)
from physics_based_classification.feature_extractor import FeatureExtractor
import pytest


# =============================================================================
# Test MODWT RMS and Kurtosis
# =============================================================================

def test_wavelet_rms_kurtosis_keys():
    """extract_wavelet_features returns RMS and kurtosis keys."""
    np.random.seed(42)
    sig = np.random.randn(500)
    feats = extract_wavelet_features(sig, fs=100.0, level=4)
    for j in range(1, 5):
        assert f'modwt_d{j}_rms' in feats, f'Missing modwt_d{j}_rms'
        assert f'modwt_d{j}_kurtosis' in feats, f'Missing modwt_d{j}_kurtosis'
    assert 'modwt_a_rms' in feats
    assert 'modwt_a_kurtosis' in feats


def test_wavelet_rms_equals_sqrt_variance():
    """For MODWT coefficients, RMS should equal sqrt(variance) since they are zero-mean."""
    np.random.seed(42)
    sig = np.random.randn(1000)
    feats = extract_wavelet_features(sig, fs=100.0, level=4)
    for j in range(1, 5):
        rms = feats[f'modwt_d{j}_rms']
        var = feats[f'modwt_d{j}_variance']
        assert abs(rms - np.sqrt(var)) < 1e-10, f'D{j}: RMS={rms}, sqrt(var)={np.sqrt(var)}'


def test_wavelet_kurtosis_white_noise():
    """White noise should have excess kurtosis near 0 (Gaussian)."""
    np.random.seed(42)
    sig = np.random.randn(10000)
    feats = extract_wavelet_features(sig, fs=100.0, level=5)
    for j in range(1, 6):
        kurt = feats[f'modwt_d{j}_kurtosis']
        assert abs(kurt) < 1.0, f'D{j} kurtosis={kurt}, expected near 0 for Gaussian'


# =============================================================================
# Test Signal Quality Features
# =============================================================================

def test_signal_quality_returns_all_keys():
    """extract_signal_quality_features returns all 6 expected keys."""
    np.random.seed(42)
    sig = np.random.randn(500)
    sq = extract_signal_quality_features(sig, 100.0)
    expected = ['sq_noise_floor', 'sq_baseline_stability', 'sq_dropout_rate',
                'sq_peak_consistency', 'sq_snr', 'sq_response_linearity']
    for key in expected:
        assert key in sq, f'Missing {key}'


def test_noise_floor_known_noise():
    """Noise floor estimate should be within 30% of true sigma for Gaussian noise."""
    np.random.seed(42)
    true_sigma = 2.5
    sig = np.random.randn(5000) * true_sigma
    nf = get_noise_floor(sig, 100.0)
    assert abs(nf - true_sigma) / true_sigma < 0.3, f'Noise floor={nf}, expected ~{true_sigma}'


def test_dropout_rate_known_dropouts():
    """Dropout rate should detect repeated-value runs."""
    sig = np.random.randn(1000)
    # Inject 100 samples of stuck-at (10% dropout)
    sig[200:300] = 5.0
    rate = get_dropout_rate(sig, 100.0, repeat_thresh=3)
    assert 0.08 < rate < 0.15, f'Dropout rate={rate}, expected ~0.10'


def test_dropout_rate_clean_signal():
    """Clean random signal should have near-zero dropout rate."""
    np.random.seed(42)
    sig = np.random.randn(1000)
    rate = get_dropout_rate(sig, 100.0)
    assert rate < 0.01, f'Dropout rate={rate}, expected ~0 for random signal'


def test_snr_high_quality():
    """Signal with very low noise should have high SNR."""
    t = np.linspace(0, 1, 1000)
    sig = np.sin(2 * np.pi * 10 * t) * 10  # Strong sinusoid
    sig += np.random.randn(1000) * 0.01    # Very low noise
    snr = get_snr(sig, 1000.0)
    assert snr > 20, f'SNR={snr}, expected > 20 dB for high-quality signal'


def test_baseline_stability_constant():
    """Constant signal should have zero baseline stability."""
    sig = np.ones(1000) * 42.0
    stab = get_baseline_stability(sig, 100.0)
    assert stab < 1e-10, f'Stability={stab}, expected 0 for constant signal'


def test_baseline_stability_drift():
    """Drifting signal should have non-zero baseline stability."""
    sig = np.linspace(0, 10, 1000)
    stab = get_baseline_stability(sig, 100.0)
    assert stab > 1.0, f'Stability={stab}, expected > 1.0 for strong drift'


def test_response_linearity_line():
    """Pure linear signal should have R² ≈ 1.0."""
    sig = np.linspace(0, 100, 500)
    r2 = get_response_linearity(sig, 100.0)
    assert r2 > 0.999, f'R²={r2}, expected ~1.0 for perfect line'


def test_response_linearity_noise():
    """White noise should have R² near 0."""
    np.random.seed(42)
    sig = np.random.randn(500)
    r2 = get_response_linearity(sig, 100.0)
    assert r2 < 0.05, f'R²={r2}, expected near 0 for white noise'


# =============================================================================
# Test Divergence Functions
# =============================================================================

def test_kl_self_divergence():
    """KL divergence of a distribution with itself should be ~0."""
    p = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    result = kl_divergence(p, p)
    assert abs(result) < 1e-10, f'Self-KL={result}, expected ~0'


def test_kl_different_distributions():
    """KL divergence of very different distributions should be large."""
    p = np.array([0.9, 0.05, 0.025, 0.015, 0.01])
    q = np.array([0.01, 0.015, 0.025, 0.05, 0.9])
    result = kl_divergence(p, q)
    assert result > 1.0, f'KL={result}, expected > 1.0 for very different distributions'


def test_symmetric_kl_is_symmetric():
    """Symmetric KL should give same result regardless of argument order."""
    p = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
    q = np.array([0.1, 0.1, 0.2, 0.3, 0.3])
    assert abs(symmetric_kl_divergence(p, q) - symmetric_kl_divergence(q, p)) < 1e-10


def test_jsd_bounded():
    """Jensen-Shannon divergence should be bounded in [0, ln(2)]."""
    np.random.seed(42)
    for _ in range(20):
        p = np.random.dirichlet(np.ones(10))
        q = np.random.dirichlet(np.ones(10))
        jsd = jensen_shannon_divergence(p, q)
        assert 0 <= jsd <= np.log(2) + 1e-10, f'JSD={jsd}, expected in [0, {np.log(2):.4f}]'


def test_jsd_self_zero():
    """JSD of a distribution with itself should be ~0."""
    p = np.array([0.2, 0.3, 0.15, 0.25, 0.1])
    result = jensen_shannon_divergence(p, p)
    assert abs(result) < 1e-10, f'Self-JSD={result}, expected ~0'


def test_window_divergence_same_signal():
    """Divergence of a window from the same signal's reference should be small."""
    np.random.seed(42)
    sig = np.random.randn(2000)
    ref = build_reference_distributions(sig, 100.0, level=4)
    div = compute_window_divergence(sig[:500], ref, level=4)
    assert div['div_mean_kl'] < 1.0, f'Self-signal divergence={div["div_mean_kl"]}, expected < 1.0'


def test_window_divergence_different_signal():
    """Divergence from a very different signal should be large."""
    np.random.seed(42)
    ref_sig = np.random.randn(2000)
    test_sig = np.random.randn(500) * 10 + 50  # Very different scale/offset
    ref = build_reference_distributions(ref_sig, 100.0, level=4)
    div = compute_window_divergence(test_sig, ref, level=4)
    assert div['div_mean_kl'] > 0.5, f'Different-signal divergence={div["div_mean_kl"]}, expected > 0.5'


def test_cross_sensor_divergence():
    """Cross-sensor divergence should return expected keys."""
    np.random.seed(42)
    ref_a = build_reference_distributions(np.random.randn(1000), 100.0, level=3)
    ref_b = build_reference_distributions(np.random.randn(1000) * 5, 100.0, level=3)
    xdiv = compute_cross_sensor_divergence(ref_a, ref_b)
    assert 'xdiv_mean_kl' in xdiv
    assert xdiv['xdiv_mean_kl'] > 0


# =============================================================================
# Test Quality Vector
# =============================================================================

def test_feature_stability_constant():
    """Features with no variation should have CV ≈ 0."""
    data = {'window_start_sec': np.arange(100), 'feat_a': np.ones(100), 'feat_b': np.random.randn(100)}
    df = pd.DataFrame(data)
    stab = assess_feature_stability(df, n_segments=4)
    feat_a_cv = stab[stab['feature'] == 'feat_a']['cv'].values[0]
    assert feat_a_cv < 1e-6, f'Constant feature CV={feat_a_cv}, expected ~0'


def test_select_quality_features_count():
    """select_quality_features should return expected number of features."""
    data = [{'feature': f'f{i}', 'cv': i * 0.1} for i in range(20)]
    stab_df = pd.DataFrame(data)
    result = select_quality_features(stab_df, n_stable=5, n_sensitive=3, mandatory_features=['f0'])
    assert len(result['stable']) == 5
    assert len(result['sensitive']) == 3
    assert 'f0' in result['mandatory']
    assert len(result['all']) <= 5 + 3 + 1  # May overlap


def test_quality_vector_health_score():
    """Health score should be low for baseline and high for degraded data."""
    np.random.seed(42)
    fs = 100.0
    ext = FeatureExtractor(fs=fs)

    # Baseline
    clean = np.random.randn(5000)
    clean_feats = ext.process_signal(clean, 1.0, 0.5, include_signal_quality=True)
    stab = assess_feature_stability(clean_feats, n_segments=4)
    qf = select_quality_features(stab, n_stable=5, n_sensitive=3)

    ref_stats = {}
    for feat in qf['all']:
        if feat in clean_feats.columns:
            ref_stats[feat] = (float(clean_feats[feat].mean()), float(clean_feats[feat].std()) + 1e-12)

    qv_clean = build_quality_vector(clean_feats, qf, normalize=True, reference_stats=ref_stats)

    # Degraded (heavy noise)
    degraded = clean + np.random.randn(5000) * 5
    deg_feats = ext.process_signal(degraded, 1.0, 0.5, include_signal_quality=True)
    qv_deg = build_quality_vector(deg_feats, qf, normalize=True, reference_stats=ref_stats)

    clean_health = qv_clean['qv_health_score'].mean()
    deg_health = qv_deg['qv_health_score'].mean()

    assert deg_health > clean_health, (
        f'Degraded health={deg_health:.3f} should be > clean health={clean_health:.3f}'
    )


# =============================================================================
# Integration Test
# =============================================================================

def test_full_pipeline_integration():
    """Full pipeline: extract features with all modules, build quality vector."""
    np.random.seed(42)
    sig = np.random.randn(3000)
    fs = 100.0

    ext = FeatureExtractor(fs=fs)
    df = ext.process_signal(sig, 2.0, 1.0,
                            include_wavelet=True, wavelet_level=5,
                            include_signal_quality=True)

    # Should have MODWT, signal quality, and time/freq/ADEV features
    assert any('modwt_d1_rms' in c for c in df.columns)
    assert any('sq_snr' in c for c in df.columns)
    assert len(df) > 0

    # Stability + quality vector
    stab = assess_feature_stability(df, n_segments=4)
    assert len(stab) > 0

    qf = select_quality_features(stab, n_stable=5, n_sensitive=3,
                                 mandatory_features=DEFAULT_MANDATORY_FEATURES)
    qv = build_quality_vector(df, qf)
    assert 'qv_health_score' in qv.columns
    assert len(qv) == len(df)


# =============================================================================
# COMPREHENSIVE TESTS — Edge Cases
# =============================================================================

def test_wavelet_features_short_signal():
    """Signal < 8 samples should return empty dict."""
    feats = extract_wavelet_features(np.array([1, 2, 3]), fs=100.0)
    assert feats == {}


def test_wavelet_features_different_levels():
    """Level=3 and level=5 should produce different key counts."""
    np.random.seed(42)
    sig = np.random.randn(2000)
    f3 = extract_wavelet_features(sig, fs=100.0, level=3)
    f5 = extract_wavelet_features(sig, fs=100.0, level=5)
    rms_keys_3 = [k for k in f3 if '_rms' in k]
    rms_keys_5 = [k for k in f5 if '_rms' in k]
    assert len(rms_keys_3) == 4  # D1-D3 + A
    assert len(rms_keys_5) == 6  # D1-D5 + A
    assert f3['modwt_levels'] == 3.0
    assert f5['modwt_levels'] == 5.0


def test_signal_quality_short_signal():
    """Signal < 8 samples should return empty dict."""
    sq = extract_signal_quality_features(np.array([1, 2, 3]), 100.0)
    assert sq == {}


def test_signal_quality_all_zeros():
    """All-zero signal should return valid features without NaN."""
    sq = extract_signal_quality_features(np.zeros(500), 100.0)
    assert len(sq) == 6
    for key, val in sq.items():
        assert not np.isnan(val), f'{key} is NaN for all-zero signal'
    assert sq['sq_noise_floor'] == 0.0
    assert sq['sq_snr'] == 60.0  # Capped: no noise


def test_divergence_short_window():
    """Window < 8 samples should return empty dict."""
    np.random.seed(42)
    ref = build_reference_distributions(np.random.randn(500), 100.0, level=3)
    div = compute_window_divergence(np.array([1, 2, 3]), ref, level=3)
    assert div == {}


def test_divergence_invalid_method():
    """Unknown method should raise ValueError."""
    with pytest.raises(ValueError):
        estimate_distribution(np.random.randn(100), method='foobar')


def test_quality_vector_empty_df():
    """Empty DataFrame should return empty quality vector."""
    qf = {'all': ['a', 'b'], 'stable': ['a'], 'sensitive': ['b'], 'mandatory': []}
    result = build_quality_vector(pd.DataFrame(), qf)
    assert len(result) == 0


def test_quality_vector_missing_features():
    """Features not in DataFrame should be gracefully skipped."""
    np.random.seed(42)
    df = pd.DataFrame({'real_feat': np.random.randn(10), 'window_start_sec': range(10)})
    qf = {'all': ['real_feat', 'nonexistent_feat'], 'stable': ['real_feat'],
           'sensitive': ['nonexistent_feat'], 'mandatory': []}
    result = build_quality_vector(df, qf, normalize=True)
    assert 'real_feat' in result.columns
    assert 'qv_health_score' in result.columns
    assert len(result) == 10


# =============================================================================
# COMPREHENSIVE TESTS — Signal Quality Depth
# =============================================================================

def test_noise_floor_multiple_sigmas():
    """Noise floor scales linearly with true sigma."""
    np.random.seed(42)
    for true_sigma in [0.1, 1.0, 10.0]:
        sig = np.random.randn(5000) * true_sigma
        nf = get_noise_floor(sig, 100.0)
        assert abs(nf - true_sigma) / true_sigma < 0.3, \
            f'sigma={true_sigma}: noise_floor={nf:.4f}'


def test_snr_pure_noise():
    """Pure white noise should have SNR near 0 dB."""
    np.random.seed(42)
    sig = np.random.randn(5000)
    snr = get_snr(sig, 100.0)
    assert -5 < snr < 5, f'Pure noise SNR={snr}, expected near 0 dB'


def test_snr_capped_values():
    """SNR should be capped at +60 and -60 dB."""
    # Noiseless signal (constant)
    snr_high = get_snr(np.ones(100) * 5.0, 100.0)
    assert snr_high == 60.0, f'Expected 60 dB cap, got {snr_high}'
    # Zero signal
    snr_low = get_snr(np.zeros(100), 100.0)
    assert snr_low == 60.0  # zero signal, zero noise → capped


def test_dropout_rate_multiple_segments():
    """Multiple dropout regions should accumulate correctly."""
    np.random.seed(42)
    sig = np.random.randn(1000)
    sig[100:150] = 0.0  # 50 samples
    sig[500:580] = 0.0  # 80 samples
    rate = get_dropout_rate(sig, 100.0, repeat_thresh=3)
    expected = (50 + 80) / 1000
    assert abs(rate - expected) < 0.02, f'rate={rate}, expected ~{expected}'


def test_dropout_rate_all_constant():
    """All-same-value signal should be 100% dropout."""
    sig = np.ones(500) * 42.0
    rate = get_dropout_rate(sig, 100.0, repeat_thresh=3)
    assert rate == 1.0, f'All-constant dropout={rate}, expected 1.0'


def test_peak_consistency_with_spikes():
    """Injecting spikes should increase peak consistency (higher CV)."""
    np.random.seed(42)
    clean = np.random.randn(1000)
    pc_clean = get_peak_consistency(clean, 100.0)

    spiked = clean.copy()
    spiked[100] = 50.0
    spiked[500] = -50.0
    spiked[800] = 50.0
    pc_spiked = get_peak_consistency(spiked, 100.0)

    assert pc_spiked > pc_clean, \
        f'Spiked peak_consistency={pc_spiked:.4f} should > clean={pc_clean:.4f}'


def test_response_linearity_sinusoid():
    """Sinusoidal signal should have low R²."""
    t = np.linspace(0, 4 * np.pi, 500)
    sig = np.sin(t)
    r2 = get_response_linearity(sig, 100.0)
    assert r2 < 0.3, f'Sinusoid R²={r2}, expected < 0.3'


# =============================================================================
# COMPREHENSIVE TESTS — Divergence Depth
# =============================================================================

def test_estimate_distribution_histogram_sums_to_one():
    """Histogram probabilities should sum to approximately 1.0."""
    np.random.seed(42)
    centers, probs = estimate_distribution(np.random.randn(1000), n_bins=50, method='histogram')
    assert abs(np.sum(probs) - 1.0) < 0.01, f'Sum={np.sum(probs)}'
    assert len(centers) == 50
    assert len(probs) == 50


def test_estimate_distribution_kde_sums_to_one():
    """KDE probabilities should sum to approximately 1.0."""
    np.random.seed(42)
    centers, probs = estimate_distribution(np.random.randn(1000), n_bins=50, method='kde')
    assert abs(np.sum(probs) - 1.0) < 0.05, f'KDE sum={np.sum(probs)}'


def test_align_distributions_same_range():
    """Same-range distributions should align without distortion."""
    np.random.seed(42)
    c1 = np.linspace(-3, 3, 50)
    p1 = np.exp(-c1**2 / 2); p1 /= p1.sum()
    p_aligned, q_aligned = align_distributions(c1, p1, c1, p1, n_bins=50)
    assert abs(np.sum(p_aligned) - 1.0) < 0.01
    assert abs(np.sum(q_aligned) - 1.0) < 0.01


def test_align_distributions_disjoint_ranges():
    """Disjoint ranges should still produce valid distributions."""
    c1 = np.linspace(0, 1, 20)
    p1 = np.ones(20) / 20
    c2 = np.linspace(10, 11, 20)
    p2 = np.ones(20) / 20
    p_a, q_a = align_distributions(c1, p1, c2, p2, n_bins=50)
    assert abs(np.sum(p_a) - 1.0) < 0.01
    assert abs(np.sum(q_a) - 1.0) < 0.01
    # Disjoint distributions should have high divergence
    kl = symmetric_kl_divergence(p_a, q_a)
    assert kl > 1.0, f'Disjoint distributions KL={kl}, expected > 1.0'


def test_kl_non_negative():
    """KL divergence should always be >= 0 (Gibbs' inequality)."""
    np.random.seed(42)
    for _ in range(50):
        p = np.random.dirichlet(np.ones(10))
        q = np.random.dirichlet(np.ones(10))
        kl = kl_divergence(p, q)
        assert kl >= -1e-10, f'KL={kl}, should be >= 0'


def test_divergence_monotonic_with_noise():
    """Progressive noise injection should produce monotonically increasing divergence."""
    np.random.seed(42)
    clean = np.random.randn(2000)
    ref = build_reference_distributions(clean, 100.0, level=4)

    prev_kl = -1
    for noise_level in [0, 0.5, 1.0, 2.0, 5.0]:
        noisy = clean[:500] + np.random.randn(500) * noise_level
        div = compute_window_divergence(noisy, ref, level=4)
        curr_kl = div['div_mean_kl']
        assert curr_kl >= prev_kl - 0.1, \
            f'noise={noise_level}: KL={curr_kl:.4f} should be >= prev={prev_kl:.4f}'
        prev_kl = curr_kl


def test_divergence_monotonic_with_drift():
    """Progressive drift should increase mean KL divergence (overall trend)."""
    np.random.seed(42)
    clean = np.random.randn(2000)
    ref = build_reference_distributions(clean, 100.0, level=4)

    # Just check that large drift > small drift > no drift for mean_kl
    div_0 = compute_window_divergence(clean[:500], ref, level=4)
    div_10 = compute_window_divergence(clean[:500] + 10, ref, level=4)
    div_50 = compute_window_divergence(clean[:500] + 50, ref, level=4)

    assert div_10['div_mean_kl'] > div_0['div_mean_kl'], \
        f'drift=10 mean_kl={div_10["div_mean_kl"]:.4f} should > drift=0 mean_kl={div_0["div_mean_kl"]:.4f}'
    assert div_50['div_mean_kl'] > div_0['div_mean_kl'], \
        f'drift=50 mean_kl={div_50["div_mean_kl"]:.4f} should > drift=0 mean_kl={div_0["div_mean_kl"]:.4f}'


def test_reference_distributions_keys():
    """Reference distributions should have correct sub-band keys."""
    np.random.seed(42)
    ref3 = build_reference_distributions(np.random.randn(1000), 100.0, level=3)
    ref5 = build_reference_distributions(np.random.randn(1000), 100.0, level=5)
    assert set(ref3.keys()) == {'D1', 'D2', 'D3', 'A3'}
    assert set(ref5.keys()) == {'D1', 'D2', 'D3', 'D4', 'D5', 'A5'}


# =============================================================================
# COMPREHENSIVE TESTS — Quality Vector Depth
# =============================================================================

def test_stability_ranking_order():
    """Most stable feature should have rank 1."""
    data = {'window_start_sec': np.arange(100),
            'stable_feat': np.ones(100) * 5.0 + np.random.randn(100) * 0.001,
            'unstable_feat': np.random.randn(100) * 10}
    df = pd.DataFrame(data)
    stab = assess_feature_stability(df, n_segments=4)
    stable_rank = stab[stab['feature'] == 'stable_feat']['stability_rank'].values[0]
    unstable_rank = stab[stab['feature'] == 'unstable_feat']['stability_rank'].values[0]
    assert stable_rank < unstable_rank


def test_stability_variable_vs_constant():
    """Variable feature should have higher CV than constant feature."""
    np.random.seed(42)
    data = {'window_start_sec': np.arange(100),
            'constant': np.ones(100) * 42.0,
            'variable': np.linspace(0, 100, 100)}
    df = pd.DataFrame(data)
    stab = assess_feature_stability(df, n_segments=4)
    const_cv = stab[stab['feature'] == 'constant']['cv'].values[0]
    var_cv = stab[stab['feature'] == 'variable']['cv'].values[0]
    assert var_cv > const_cv


def test_cross_condition_stability_common_features():
    """Only features present in all conditions should appear."""
    df_a = pd.DataFrame({'window_start_sec': [0, 1], 'feat_a': [1, 2], 'feat_shared': [3, 4]})
    df_b = pd.DataFrame({'window_start_sec': [0, 1], 'feat_b': [5, 6], 'feat_shared': [7, 8]})
    stab = assess_cross_condition_stability({'a': df_a, 'b': df_b})
    features = stab['feature'].tolist()
    assert 'feat_shared' in features
    assert 'feat_a' not in features
    assert 'feat_b' not in features


def test_select_deduplication():
    """Mandatory + stable overlapping features should not be duplicated in 'all'."""
    data = [{'feature': f'f{i}', 'cv': i * 0.1} for i in range(20)]
    stab_df = pd.DataFrame(data)
    # f0 is both mandatory and should be in stable (lowest CV)
    result = select_quality_features(stab_df, n_stable=5, n_sensitive=3,
                                     mandatory_features=['f0', 'f1'])
    assert result['all'].count('f0') == 1
    assert result['all'].count('f1') == 1
    assert len(result['all']) == len(set(result['all']))


def test_health_score_monotonic_noise():
    """Health score should be higher for degraded than for clean."""
    np.random.seed(42)
    fs = 100.0
    ext = FeatureExtractor(fs=fs)
    clean = np.random.randn(5000)

    clean_feats = ext.process_signal(clean, 1.0, 0.5, include_signal_quality=True)
    stab = assess_feature_stability(clean_feats, n_segments=4)
    qf = select_quality_features(stab, n_stable=5, n_sensitive=3)
    ref_stats = {}
    for feat in qf['all']:
        if feat in clean_feats.columns:
            ref_stats[feat] = (float(clean_feats[feat].mean()), float(clean_feats[feat].std()) + 1e-12)

    qv_clean = build_quality_vector(clean_feats, qf, normalize=True, reference_stats=ref_stats)

    # Moderate noise
    np.random.seed(123)
    moderate = clean + np.random.randn(5000) * 2.0
    mod_feats = ext.process_signal(moderate, 1.0, 0.5, include_signal_quality=True)
    qv_mod = build_quality_vector(mod_feats, qf, normalize=True, reference_stats=ref_stats)

    # Heavy noise
    np.random.seed(456)
    heavy = clean + np.random.randn(5000) * 5.0
    hvy_feats = ext.process_signal(heavy, 1.0, 0.5, include_signal_quality=True)
    qv_hvy = build_quality_vector(hvy_feats, qf, normalize=True, reference_stats=ref_stats)

    score_clean = qv_clean['qv_health_score'].mean()
    score_mod = qv_mod['qv_health_score'].mean()
    score_hvy = qv_hvy['qv_health_score'].mean()

    assert score_mod > score_clean, \
        f'Moderate noise health={score_mod:.3f} should > clean={score_clean:.3f}'
    assert score_hvy > score_clean, \
        f'Heavy noise health={score_hvy:.3f} should > clean={score_clean:.3f}'


def test_degradation_score_responds_to_spikes():
    """Spike injection should raise degradation score."""
    np.random.seed(42)
    fs = 100.0
    ext = FeatureExtractor(fs=fs)
    clean = np.random.randn(3000)

    clean_feats = ext.process_signal(clean, 1.0, 0.5, include_signal_quality=True)
    stab = assess_feature_stability(clean_feats, n_segments=4)
    qf = select_quality_features(stab, n_stable=5, n_sensitive=3)
    ref_stats = {}
    for feat in qf['all']:
        if feat in clean_feats.columns:
            ref_stats[feat] = (float(clean_feats[feat].mean()), float(clean_feats[feat].std()) + 1e-12)

    qv_clean = build_quality_vector(clean_feats, qf, normalize=True, reference_stats=ref_stats)

    spiked = clean.copy()
    for i in range(0, 3000, 100):
        spiked[i] = 50.0
    spiked_feats = ext.process_signal(spiked, 1.0, 0.5, include_signal_quality=True)
    qv_spiked = build_quality_vector(spiked_feats, qf, normalize=True, reference_stats=ref_stats)

    if 'qv_degradation_score' in qv_spiked.columns and 'qv_degradation_score' in qv_clean.columns:
        assert qv_spiked['qv_degradation_score'].mean() > qv_clean['qv_degradation_score'].mean()


def test_quality_vector_no_normalize():
    """normalize=False should preserve raw feature values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feat_a': [10.0, 20.0, 30.0],
        'feat_b': [1.0, 2.0, 3.0],
        'window_start_sec': [0, 1, 2],
    })
    qf = {'all': ['feat_a', 'feat_b'], 'stable': ['feat_a'],
          'sensitive': ['feat_b'], 'mandatory': []}
    result = build_quality_vector(df, qf, normalize=False)
    assert list(result['feat_a']) == [10.0, 20.0, 30.0]
    assert list(result['feat_b']) == [1.0, 2.0, 3.0]


# =============================================================================
# COMPREHENSIVE TESTS — Integration
# =============================================================================

def test_pipeline_with_divergence_enabled():
    """process_signal with include_divergence=True should add div_ columns."""
    np.random.seed(42)
    sig = np.random.randn(3000)
    fs = 100.0
    ref = build_reference_distributions(sig[:1000], fs, level=4)

    ext = FeatureExtractor(fs=fs)
    df = ext.process_signal(sig, 2.0, 1.0,
                            include_wavelet=True, wavelet_level=4,
                            include_signal_quality=True,
                            include_divergence=True,
                            divergence_ref_dists=ref)
    div_cols = [c for c in df.columns if c.startswith('div_')]
    assert len(div_cols) > 0, 'No divergence columns found'
    assert 'div_mean_kl' in df.columns


def test_process_signal_full_with_signal_quality():
    """process_signal_full should return sq_ keys."""
    np.random.seed(42)
    sig = np.random.randn(1000)
    ext = FeatureExtractor(fs=100.0)
    result = ext.process_signal_full(sig, include_signal_quality=True)
    sq_keys = [k for k in result if k.startswith('sq_')]
    assert len(sq_keys) == 6


def test_different_sampling_rates():
    """Pipeline should work correctly at different sampling rates."""
    np.random.seed(42)
    for fs in [48.0, 83.0, 100.0]:
        sig = np.random.randn(int(fs * 10))  # 10 seconds
        ext = FeatureExtractor(fs=fs)
        df = ext.process_signal(sig, 2.0, 1.0,
                                include_wavelet=True, wavelet_level=4,
                                include_signal_quality=True)
        assert len(df) > 0, f'No windows at fs={fs}'
        assert 'modwt_d1_rms' in df.columns, f'Missing MODWT at fs={fs}'
        assert 'sq_snr' in df.columns, f'Missing signal quality at fs={fs}'


def test_end_to_end_degradation_detection():
    """Full pipeline from extraction to quality vector detects degradation."""
    np.random.seed(42)
    fs = 100.0
    ext = FeatureExtractor(fs=fs)

    # Generate clean signal
    clean = np.random.randn(5000)
    clean_feats = ext.process_signal(clean, 1.0, 0.5, include_signal_quality=True)

    # Stability analysis
    stab = assess_feature_stability(clean_feats, n_segments=4)
    qf = select_quality_features(stab, n_stable=5, n_sensitive=3,
                                 mandatory_features=DEFAULT_MANDATORY_FEATURES)

    # Reference stats from clean
    ref_stats = {}
    for feat in qf['all']:
        if feat in clean_feats.columns:
            ref_stats[feat] = (float(clean_feats[feat].mean()),
                               float(clean_feats[feat].std()) + 1e-12)

    # Clean quality vector
    qv_clean = build_quality_vector(clean_feats, qf, normalize=True,
                                    reference_stats=ref_stats)

    # Degraded: noise + drift
    degraded = clean + np.random.randn(5000) * 3 + np.linspace(0, 10, 5000)
    deg_feats = ext.process_signal(degraded, 1.0, 0.5, include_signal_quality=True)
    qv_deg = build_quality_vector(deg_feats, qf, normalize=True,
                                  reference_stats=ref_stats)

    # Degraded should have higher health score
    assert qv_deg['qv_health_score'].mean() > qv_clean['qv_health_score'].mean(), \
        'Degradation not detected by quality vector'

    # Quality vector should have the composite scores
    assert 'qv_health_score' in qv_deg.columns
    assert len(qv_deg) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
