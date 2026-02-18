"""
Project SHIELD - Quality Vector Construction
=============================================
Identifies stable vs sensitive features across conditions and constructs
a compact quality vector for sensor health prediction.

The quality vector is a reduced-dimension representation of sensor health,
built from features that are:
  1. STABLE across normal operating conditions (low CV across time segments)
  2. SENSITIVE to degradation (high CV or high divergence response)

Inspired by the DLR Health Index Framework (Kamtsiuris et al., 2022)
which uses hi = f(current_params, reference_params) to express health.

This module operates on feature DataFrames produced by FeatureExtractor
and divergence DataFrames produced by divergence_analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Feature Stability Assessment
# =============================================================================

def assess_feature_stability(
    feature_df: pd.DataFrame,
    window_col: str = 'window_start_sec',
    n_segments: int = 4,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Assess the temporal stability of each feature across time segments.

    Splits the feature DataFrame into n_segments temporal chunks and
    computes the coefficient of variation (CV) of each feature's mean
    across segments. CV normalizes by magnitude so features at different
    scales are comparable.

    Parameters
    ----------
    feature_df : DataFrame from FeatureExtractor.process_signal()
    window_col : column to use for temporal ordering
    n_segments : number of temporal segments to compare
    exclude_cols : columns to exclude (metadata cols)

    Returns
    -------
    DataFrame with columns: [feature, mean, std, cv, stability_rank]
    Sorted by CV ascending (most stable first).
    """
    if exclude_cols is None:
        exclude_cols = ['window_start_sample', 'window_start_sec',
                        'signal_id', 'label']

    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    df_sorted = feature_df.sort_values(window_col).reset_index(drop=True)
    segment_size = len(df_sorted) // n_segments

    if segment_size < 1:
        return pd.DataFrame()

    segment_means = []
    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size if i < n_segments - 1 else len(df_sorted)
        seg = df_sorted.iloc[start:end]
        segment_means.append(seg[feature_cols].mean())

    seg_df = pd.DataFrame(segment_means)

    results = []
    for col in feature_cols:
        col_mean = seg_df[col].mean()
        col_std = seg_df[col].std()
        cv = col_std / (abs(col_mean) + 1e-12)
        results.append({
            'feature': col,
            'mean': col_mean,
            'std': col_std,
            'cv': cv,
        })

    results_df = pd.DataFrame(results).sort_values('cv').reset_index(drop=True)
    results_df['stability_rank'] = range(1, len(results_df) + 1)

    return results_df


def assess_cross_condition_stability(
    feature_dfs: Dict[str, pd.DataFrame],
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Assess feature stability across multiple conditions/sensors/datasets.

    Parameters
    ----------
    feature_dfs : dict mapping condition_name -> feature DataFrame

    Returns
    -------
    DataFrame with columns: [feature, mean_across_conditions,
        std_across_conditions, cv_across_conditions, n_conditions,
        stability_rank]
    """
    if exclude_cols is None:
        exclude_cols = ['window_start_sample', 'window_start_sec',
                        'signal_id', 'label']

    condition_means = {}
    common_features = None

    for name, df in feature_dfs.items():
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        if common_features is None:
            common_features = set(feature_cols)
        else:
            common_features &= set(feature_cols)
        condition_means[name] = df[feature_cols].mean()

    if not common_features:
        return pd.DataFrame()

    common_features = sorted(common_features)

    results = []
    for feat in common_features:
        values = [condition_means[name][feat] for name in feature_dfs]
        values = np.array(values)
        feat_mean = np.mean(values)
        feat_std = np.std(values)
        cv = feat_std / (abs(feat_mean) + 1e-12)
        results.append({
            'feature': feat,
            'mean_across_conditions': feat_mean,
            'std_across_conditions': feat_std,
            'cv_across_conditions': cv,
            'n_conditions': len(values),
        })

    results_df = pd.DataFrame(results).sort_values(
        'cv_across_conditions'
    ).reset_index(drop=True)
    results_df['stability_rank'] = range(1, len(results_df) + 1)

    return results_df


# =============================================================================
# Quality Vector Construction
# =============================================================================

def select_quality_features(
    stability_df: pd.DataFrame,
    n_stable: int = 8,
    n_sensitive: int = 4,
    cv_col: str = 'cv',
    mandatory_features: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Select features for the quality vector based on stability analysis.

    Parameters
    ----------
    stability_df : output of assess_feature_stability() or
                   assess_cross_condition_stability()
    n_stable : number of most-stable features to include
    n_sensitive : number of most-sensitive features to include
    cv_col : column name for coefficient of variation
    mandatory_features : features to always include regardless of ranking

    Returns
    -------
    dict with keys:
        'stable': list of stable feature names
        'sensitive': list of sensitive feature names
        'mandatory': list of mandatory feature names
        'all': combined list (deduplicated, ordered)
    """
    sorted_df = stability_df.sort_values(cv_col).reset_index(drop=True)

    stable = sorted_df.head(n_stable)['feature'].tolist()
    sensitive = sorted_df.tail(n_sensitive)['feature'].tolist()

    mandatory = mandatory_features or []

    # Combine and deduplicate while preserving order
    all_features = []
    seen = set()
    for feat in mandatory + stable + sensitive:
        if feat not in seen:
            all_features.append(feat)
            seen.add(feat)

    return {
        'stable': stable,
        'sensitive': sensitive,
        'mandatory': mandatory,
        'all': all_features,
    }


def build_quality_vector(
    feature_df: pd.DataFrame,
    quality_features: Dict[str, List[str]],
    normalize: bool = True,
    reference_stats: Optional[Dict[str, Tuple[float, float]]] = None,
) -> pd.DataFrame:
    """Build the quality vector from a feature DataFrame.

    Parameters
    ----------
    feature_df : full feature DataFrame from FeatureExtractor
    quality_features : output of select_quality_features()
    normalize : whether to z-score normalize each feature
    reference_stats : optional dict of {feature: (mean, std)} for
                      normalization. If None and normalize=True,
                      computes from the input DataFrame.

    Returns
    -------
    DataFrame with quality vector columns + metadata.
    Columns include 'qv_health_score' (composite scalar health metric)
    and 'qv_degradation_score' (early warning metric).
    """
    all_feats = quality_features['all']
    available = [f for f in all_feats if f in feature_df.columns]

    if not available:
        return pd.DataFrame()

    qv_df = feature_df[available].copy()

    if normalize:
        if reference_stats is None:
            reference_stats = {}
            for col in available:
                reference_stats[col] = (
                    float(qv_df[col].mean()),
                    float(qv_df[col].std()) + 1e-12,
                )

        for col in available:
            if col in reference_stats:
                mean, std = reference_stats[col]
                qv_df[col] = (qv_df[col] - mean) / std

    # Preserve metadata columns
    meta_cols = ['window_start_sample', 'window_start_sec']
    for mc in meta_cols:
        if mc in feature_df.columns:
            qv_df[mc] = feature_df[mc].values

    # Health score: mean |z-score| of stable features
    # Low = healthy (close to reference baseline)
    stable_feats = [f for f in quality_features['stable'] if f in qv_df.columns]
    if stable_feats:
        qv_df['qv_health_score'] = qv_df[stable_feats].abs().mean(axis=1)

    # Degradation score: mean |z-score| of sensitive features
    # Higher = more degradation detected (early warning)
    sensitive_feats = [f for f in quality_features['sensitive'] if f in qv_df.columns]
    if sensitive_feats:
        qv_df['qv_degradation_score'] = qv_df[sensitive_feats].abs().mean(axis=1)

    return qv_df


# =============================================================================
# Recommended Default Quality Vector Features
# =============================================================================

DEFAULT_MANDATORY_FEATURES = [
    'sq_snr',
    'sq_noise_floor',
    'sq_baseline_stability',
]

DEFAULT_WAVELET_STABLE_CANDIDATES = [
    'modwt_a_energy',
    'modwt_a_rms',
    'modwt_energy_ratio_hf_lf',
    'modwt_total_energy',
    'modwt_d3_entropy',
]

DEFAULT_WAVELET_SENSITIVE_CANDIDATES = [
    'modwt_d1_energy',
    'modwt_d1_kurtosis',
    'modwt_d2_energy',
    'modwt_a_kurtosis',
]
