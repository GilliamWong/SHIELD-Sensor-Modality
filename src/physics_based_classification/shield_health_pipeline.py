"""
Project SHIELD - Healthy Envelope Validation Pipeline
=====================================================
Utilities for loading the March 2 SHIELD recollection data, fitting a
healthy-data calibration envelope, and evaluating synthetic fault response.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .divergence_analysis import (
    build_reference_distributions,
    compute_cross_sensor_divergence,
    compute_window_divergence,
)
from .fault_injection import FAULT_TYPES, inject_synthetic_fault
from .feature_extractor import FeatureExtractor
from .quality_vector import (
    DEFAULT_MANDATORY_FEATURES,
    apply_health_score_calibration,
    assess_cross_condition_stability,
    build_quality_vector,
    fit_health_score_calibration,
    select_quality_features,
)


CONDITIONS = (
    'stationary',
    '40_rpm',
    '80_rpm',
    '120_rpm',
    '160_rpm',
    '200_rpm',
)

CONDITION_LABELS = {
    'stationary': 'Stationary',
    '40_rpm': '40 RPM',
    '80_rpm': '80 RPM',
    '120_rpm': '120 RPM',
    '160_rpm': '160 RPM',
    '200_rpm': '200 RPM',
}

RPM_VALUES = {
    'stationary': 0,
    '40_rpm': 40,
    '80_rpm': 80,
    '120_rpm': 120,
    '160_rpm': 160,
    '200_rpm': 200,
}

DEFAULT_AXES = {
    'AccX': 'CalAX',
    'AccY': 'CalAY',
    'AccZ': 'CalAZ',
    'GyroX': 'CalGX',
}

RAW_AXIS_COLUMNS = {
    'AccX': 'RawAX',
    'AccY': 'RawAY',
    'AccZ': 'RawAZ',
    'GyroX': 'RawGX',
}

DEFAULT_FS = 100.0
DEFAULT_WINDOW_SEC = 3.0
DEFAULT_STEP_SEC = 1.5
DEFAULT_WAVELET_LEVEL = 5
TRIM_START_SEC = 60
TRIM_END_SEC = 30
FAULT_BASE_DURATION_SEC = 60
HEALTH_QUANTILE = 0.95
FAULT_REFERENCE_CONDITIONS = CONDITIONS[1:]

SENSOR_FILE_PATTERNS = {
    'DFRobot': {
        'stationary': 'dfrobot_stationary_test_*.csv',
        '40_rpm': 'dfrobot_40_rpm_calibrated_*.csv',
        '80_rpm': 'dfrobot_80_rpm_calibrated_*.csv',
        '120_rpm': 'dfrobot_120_rpm_calibrated_*.csv',
        '160_rpm': 'dfrobot_160_rpm_calibrated_*.csv',
        '200_rpm': 'dfrobot_200_rpm_calibrated_*.csv',
    },
    'Pololu': {
        'stationary': 'pololu_stationary_test_calibrated_*.csv',
        '40_rpm': 'pololu_40_rpm_calibrated_*.csv',
        '80_rpm': 'pololu_80_rpm_calibrated_*.csv',
        '120_rpm': 'pololu_120_rpm_calibrated_*.csv',
        '160_rpm': 'pololu_160_rpm_calibrated_*.csv',
        '200_rpm': 'pololu_200_rpm_calibrated_*.csv',
    },
}


@dataclass
class SensorHealthModel:
    sensor: str
    quality_features: Dict[str, List[str]]
    stability_df: pd.DataFrame
    reference_stats: Dict[str, tuple[float, float]]
    health_calibration: Dict[str, float]
    healthy_qv: pd.DataFrame
    condition_qv: Dict[str, pd.DataFrame]
    timeline_qv: pd.DataFrame
    accx_reference_distributions: Dict[str, tuple[np.ndarray, np.ndarray]]
    divergence_timeline: pd.DataFrame


def _find_single_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one match for pattern '{pattern}', found {len(matches)}")
    return matches[0]


def discover_shield_recollection_files(data_root: Path | str) -> Dict[str, Dict[str, Path]]:
    root = Path(data_root)
    return {
        sensor: {
            condition: _find_single_match(root, pattern)
            for condition, pattern in mapping.items()
        }
        for sensor, mapping in SENSOR_FILE_PATTERNS.items()
    }


def load_shield_recollection(data_root: Path | str) -> Dict[str, Dict[str, pd.DataFrame]]:
    files = discover_shield_recollection_files(data_root)
    loaded: Dict[str, Dict[str, pd.DataFrame]] = {}

    for sensor, mapping in files.items():
        loaded[sensor] = {}
        for condition, path in mapping.items():
            df = pd.read_csv(path)
            df = df.sort_values('Time_ms').reset_index(drop=True)
            required = {'Time_ms', *DEFAULT_AXES.values(), 'CalGY', 'CalGZ', 'RawAX', 'RawAY', 'RawAZ', 'RawGX', 'RawGY', 'RawGZ'}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(f"{path.name} missing columns: {missing}")
            loaded[sensor][condition] = df

    return loaded


def trim_signal(
    signal: np.ndarray,
    fs: float = DEFAULT_FS,
    trim_start_sec: int = TRIM_START_SEC,
    trim_end_sec: int = TRIM_END_SEC,
) -> np.ndarray:
    start = int(trim_start_sec * fs)
    end = int(trim_end_sec * fs)
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if start + end >= values.size:
        return values.copy()
    return values[start: values.size - end]


def build_trimmed_signals(
    data_by_sensor: Mapping[str, Mapping[str, pd.DataFrame]],
    axes: Mapping[str, str] = DEFAULT_AXES,
    fs: float = DEFAULT_FS,
    trim_start_sec: int = TRIM_START_SEC,
    trim_end_sec: int = TRIM_END_SEC,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    trimmed: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for sensor, condition_map in data_by_sensor.items():
        trimmed[sensor] = {}
        for condition in CONDITIONS:
            df = condition_map[condition]
            trimmed[sensor][condition] = {
                axis: trim_signal(df[column].to_numpy(dtype=np.float64), fs, trim_start_sec, trim_end_sec)
                for axis, column in axes.items()
            }

    return trimmed


def build_stationary_sanity_summary(
    data_by_sensor: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for sensor, condition_map in data_by_sensor.items():
        df = condition_map['stationary']
        cal_acc = df[['CalAX', 'CalAY', 'CalAZ']].to_numpy(dtype=np.float64)
        raw_acc = df[['RawAX', 'RawAY', 'RawAZ']].to_numpy(dtype=np.float64)
        cal_gyro = df[['CalGX', 'CalGY', 'CalGZ']].to_numpy(dtype=np.float64)
        raw_gyro = df[['RawGX', 'RawGY', 'RawGZ']].to_numpy(dtype=np.float64)

        rows.append({
            'sensor': sensor,
            'cal_acc_norm_mean': float(np.linalg.norm(cal_acc, axis=1).mean()),
            'raw_acc_norm_mean': float(np.linalg.norm(raw_acc, axis=1).mean()),
            'cal_gyro_norm_mean': float(np.linalg.norm(cal_gyro, axis=1).mean()),
            'raw_gyro_norm_mean': float(np.linalg.norm(raw_gyro, axis=1).mean()),
        })

    return pd.DataFrame(rows)


def build_modwt_segments(
    trimmed_signals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    fs: float = DEFAULT_FS,
    duration_sec: int = 10,
) -> Dict[str, np.ndarray]:
    segment_len = int(duration_sec * fs)
    return {
        sensor: signals['stationary']['AccX'][:segment_len]
        for sensor, signals in trimmed_signals.items()
    }


def _middle_slice(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if size >= values.size:
        return values.copy()
    start = max((values.size - size) // 2, 0)
    return values[start: start + size]


def build_fault_reference_axes(
    trimmed_condition_signals: Mapping[str, Mapping[str, np.ndarray]],
    fs: float = DEFAULT_FS,
    duration_sec: int = FAULT_BASE_DURATION_SEC,
    conditions: Sequence[str] = FAULT_REFERENCE_CONDITIONS,
) -> Dict[str, np.ndarray]:
    total_samples = int(duration_sec * fs)
    if total_samples <= 0:
        raise ValueError("duration_sec must produce at least one sample")

    active_conditions = [condition for condition in conditions if condition in trimmed_condition_signals]
    if not active_conditions:
        active_conditions = list(CONDITIONS)

    base_axes: Dict[str, np.ndarray] = {}
    for axis_name in DEFAULT_AXES:
        parts: List[np.ndarray] = []
        remaining = total_samples

        for idx, condition in enumerate(active_conditions):
            condition_signal = trimmed_condition_signals[condition][axis_name]
            n_left = len(active_conditions) - idx
            take = max(1, remaining // n_left)
            part = _middle_slice(condition_signal, take)
            parts.append(part)
            remaining -= len(part)

        combined = np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        if combined.size < total_samples:
            fallback = trimmed_condition_signals[active_conditions[-1]][axis_name]
            pad = _middle_slice(fallback, total_samples - combined.size)
            combined = np.concatenate([combined, pad])

        base_axes[axis_name] = combined[:total_samples]

    return base_axes


def _extract_multiaxis_feature_df(
    axis_signals: Mapping[str, np.ndarray],
    fs: float = DEFAULT_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    wavelet_level: int = DEFAULT_WAVELET_LEVEL,
) -> pd.DataFrame:
    extractor = FeatureExtractor(fs=fs)
    merged: pd.DataFrame | None = None

    for axis_name, signal in axis_signals.items():
        feat_df = extractor.process_signal(
            signal,
            window_sec,
            step_sec,
            include_adev=False,
            include_wavelet=True,
            wavelet_level=wavelet_level,
            include_signal_quality=True,
        )

        meta_cols = ['window_start_sample', 'window_start_sec']
        rename_map = {
            col: f'{axis_name}_{col}'
            for col in feat_df.columns
            if col not in meta_cols
        }
        feat_df = feat_df.rename(columns=rename_map)

        if merged is None:
            merged = feat_df
        else:
            merged = pd.concat(
                [merged, feat_df.drop(columns=[c for c in meta_cols if c in feat_df.columns])],
                axis=1,
            )

    return merged if merged is not None else pd.DataFrame()


def build_sensor_feature_maps(
    trimmed_signals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    fs: float = DEFAULT_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    wavelet_level: int = DEFAULT_WAVELET_LEVEL,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    per_sensor: Dict[str, Dict[str, pd.DataFrame]] = {}

    for sensor, condition_map in trimmed_signals.items():
        per_sensor[sensor] = {}
        for condition in CONDITIONS:
            per_sensor[sensor][condition] = _extract_multiaxis_feature_df(
                condition_map[condition],
                fs=fs,
                window_sec=window_sec,
                step_sec=step_sec,
                wavelet_level=wavelet_level,
            )

    return per_sensor


def _compute_reference_stats(feature_df: pd.DataFrame, feature_names: Iterable[str]) -> Dict[str, tuple[float, float]]:
    stats: Dict[str, tuple[float, float]] = {}
    for feature_name in feature_names:
        if feature_name in feature_df.columns:
            stats[feature_name] = (
                float(feature_df[feature_name].mean()),
                float(feature_df[feature_name].std()) + 1e-12,
            )
    return stats


def _rank_fault_responsive_features(
    base_axes: Mapping[str, np.ndarray],
    reference_stats: Mapping[str, tuple[float, float]],
    candidate_features: Sequence[str],
    fs: float = DEFAULT_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    wavelet_level: int = DEFAULT_WAVELET_LEVEL,
    fault_duration_sec: int = FAULT_BASE_DURATION_SEC,
    fault_types: Sequence[str] = FAULT_TYPES,
    severity: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    base_len = int(fault_duration_sec * fs)
    axis_names = list(DEFAULT_AXES.keys())
    stacked = np.column_stack([base_axes[axis][:base_len] for axis in axis_names])
    baseline_df = _extract_multiaxis_feature_df(
        {axis: stacked[:, idx] for idx, axis in enumerate(axis_names)},
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        wavelet_level=wavelet_level,
    )
    baseline_means = baseline_df.mean()
    rng = np.random.default_rng(seed)
    scores = {feature_name: [] for feature_name in candidate_features}

    for fault_type in fault_types:
        local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        degraded = inject_synthetic_fault(stacked, fault_type, severity=severity, rng=local_rng)
        degraded_df = _extract_multiaxis_feature_df(
            {axis: degraded[:, idx] for idx, axis in enumerate(axis_names)},
            fs=fs,
            window_sec=window_sec,
            step_sec=step_sec,
            wavelet_level=wavelet_level,
        )
        degraded_means = degraded_df.mean()

        for feature_name in candidate_features:
            if feature_name not in baseline_means or feature_name not in degraded_means:
                continue
            _, ref_std = reference_stats.get(feature_name, (0.0, 1.0))
            ref_std = max(float(ref_std), 1e-12)
            shift = abs(float(degraded_means[feature_name] - baseline_means[feature_name])) / ref_std
            scores[feature_name].append(shift)

    rows = []
    for feature_name, values in scores.items():
        rows.append({
            'feature': feature_name,
            'fault_response': float(np.mean(values)) if values else 0.0,
        })

    return pd.DataFrame(rows).sort_values('fault_response', ascending=False).reset_index(drop=True)


def _build_qv_timeline(
    sensor: str,
    per_condition_qv: Mapping[str, pd.DataFrame],
    condition_lengths: Mapping[str, int],
    fs: float = DEFAULT_FS,
) -> pd.DataFrame:
    offset_sec = 0.0
    frames = []

    for condition in CONDITIONS:
        qv = per_condition_qv[condition].copy()
        qv['sensor'] = sensor
        qv['condition'] = condition
        qv['condition_label'] = CONDITION_LABELS[condition]
        qv['rpm'] = RPM_VALUES[condition]
        qv['window_start_sec_global'] = qv['window_start_sec'] + offset_sec
        frames.append(qv)
        offset_sec += condition_lengths[condition] / fs

    return pd.concat(frames, ignore_index=True)


def build_sensor_health_model(
    sensor: str,
    per_condition_features: Mapping[str, pd.DataFrame],
    trimmed_signals: Mapping[str, Dict[str, np.ndarray]],
    fs: float = DEFAULT_FS,
    healthy_quantile: float = HEALTH_QUANTILE,
    n_stable: int = 8,
    n_sensitive: int = 4,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    wavelet_level: int = DEFAULT_WAVELET_LEVEL,
) -> SensorHealthModel:
    stability_df = assess_cross_condition_stability(per_condition_features)
    healthy_df = pd.concat([per_condition_features[condition] for condition in CONDITIONS], ignore_index=True)
    all_reference_stats = _compute_reference_stats(healthy_df, healthy_df.columns)
    mandatory_features = [
        f'{axis}_{feature_name}'
        for axis in DEFAULT_AXES
        for feature_name in DEFAULT_MANDATORY_FEATURES
    ]
    mandatory_features = [feature_name for feature_name in mandatory_features if feature_name in healthy_df.columns]

    candidate_pool = stability_df['feature'].head(48).tolist()
    fault_reference_axes = build_fault_reference_axes(
        trimmed_signals,
        fs=fs,
        duration_sec=FAULT_BASE_DURATION_SEC,
    )
    fault_response_df = _rank_fault_responsive_features(
        fault_reference_axes,
        all_reference_stats,
        candidate_pool,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        wavelet_level=wavelet_level,
    )
    stable_candidates = stability_df[stability_df['cv_across_conditions'] <= 0.35]['feature'].tolist()
    if len(stable_candidates) < n_stable:
        stable_candidates = stability_df['feature'].head(48).tolist()

    stable = [
        feature_name
        for feature_name in fault_response_df['feature']
        if feature_name in stable_candidates and feature_name not in mandatory_features
    ][:n_stable]
    sensitive = [
        feature_name
        for feature_name in fault_response_df['feature']
        if feature_name not in mandatory_features and feature_name not in stable
    ][:n_sensitive]

    quality_features = {
        'stable': stable,
        'sensitive': sensitive,
        'mandatory': mandatory_features,
        'all': list(dict.fromkeys(mandatory_features + stable + sensitive)),
    }

    if not quality_features['stable']:
        quality_features = select_quality_features(
            stability_df,
            n_stable=n_stable,
            n_sensitive=n_sensitive,
            cv_col='cv_across_conditions',
            mandatory_features=mandatory_features,
        )

    reference_stats = _compute_reference_stats(healthy_df, quality_features['all'])

    healthy_qv = build_quality_vector(
        healthy_df,
        quality_features,
        normalize=True,
        reference_stats=reference_stats,
    )
    health_calibration = fit_health_score_calibration(
        healthy_qv['qv_health_score_raw'],
        healthy_quantile=healthy_quantile,
    )
    healthy_qv = apply_health_score_calibration(healthy_qv, health_calibration)

    per_condition_qv: Dict[str, pd.DataFrame] = {}
    for condition in CONDITIONS:
        qv = build_quality_vector(
            per_condition_features[condition],
            quality_features,
            normalize=True,
            reference_stats=reference_stats,
        )
        qv = apply_health_score_calibration(qv, health_calibration)
        per_condition_qv[condition] = qv

    condition_lengths = {
        condition: len(trimmed_signals[condition]['AccX'])
        for condition in CONDITIONS
    }
    timeline_qv = _build_qv_timeline(sensor, per_condition_qv, condition_lengths, fs=fs)

    healthy_accx = np.concatenate([trimmed_signals[condition]['AccX'] for condition in CONDITIONS])
    accx_reference = build_reference_distributions(
        healthy_accx,
        fs=fs,
        level=wavelet_level,
    )
    divergence_timeline = build_divergence_timeline(
        sensor=sensor,
        per_condition_signals={condition: trimmed_signals[condition]['AccX'] for condition in CONDITIONS},
        ref_dists=accx_reference,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        level=wavelet_level,
    )

    return SensorHealthModel(
        sensor=sensor,
        quality_features=quality_features,
        stability_df=stability_df,
        reference_stats=reference_stats,
        health_calibration=health_calibration,
        healthy_qv=healthy_qv,
        condition_qv=per_condition_qv,
        timeline_qv=timeline_qv,
        accx_reference_distributions=accx_reference,
        divergence_timeline=divergence_timeline,
    )


def build_divergence_timeline(
    sensor: str,
    per_condition_signals: Mapping[str, np.ndarray],
    ref_dists: Dict[str, tuple[np.ndarray, np.ndarray]],
    fs: float = DEFAULT_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    level: int = DEFAULT_WAVELET_LEVEL,
) -> pd.DataFrame:
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    rows: List[Dict[str, float | str]] = []
    offset_sec = 0.0

    for condition in CONDITIONS:
        signal = np.asarray(per_condition_signals[condition], dtype=np.float64)
        for start in range(0, signal.size - window_samples + 1, step_samples):
            window = signal[start: start + window_samples]
            div = compute_window_divergence(window, ref_dists, level=level)
            if not div:
                continue
            rows.append({
                'sensor': sensor,
                'condition': condition,
                'condition_label': CONDITION_LABELS[condition],
                'rpm': RPM_VALUES[condition],
                'window_start_sec': start / fs,
                'window_start_sec_global': offset_sec + start / fs,
                'div_mean_kl': div.get('div_mean_kl', np.nan),
                'div_mean_jsd': div.get('div_mean_jsd', np.nan),
                'div_max_kl': div.get('div_max_kl', np.nan),
            })
        offset_sec += signal.size / fs

    return pd.DataFrame(rows)


def summarize_signal_divergence(
    signal: np.ndarray,
    ref_dists: Dict[str, tuple[np.ndarray, np.ndarray]],
    fs: float = DEFAULT_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    level: int = DEFAULT_WAVELET_LEVEL,
) -> Dict[str, float]:
    """Average divergence metrics across sliding windows of a single signal."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    kl_values: List[float] = []
    jsd_values: List[float] = []

    for start in range(0, values.size - window_samples + 1, step_samples):
        window = values[start: start + window_samples]
        div = compute_window_divergence(window, ref_dists, level=level)
        if not div:
            continue
        if 'div_mean_kl' in div:
            kl_values.append(float(div['div_mean_kl']))
        if 'div_mean_jsd' in div:
            jsd_values.append(float(div['div_mean_jsd']))

    return {
        'mean_div_mean_kl': float(np.mean(kl_values)) if kl_values else np.nan,
        'mean_div_mean_jsd': float(np.mean(jsd_values)) if jsd_values else np.nan,
    }


def build_cross_sensor_divergence(
    models: Mapping[str, SensorHealthModel],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sensors = list(models.keys())
    matrix = pd.DataFrame(index=sensors, columns=sensors, dtype=float)
    band_rows: List[Dict[str, float | str]] = []

    for sensor_a in sensors:
        for sensor_b in sensors:
            if sensor_a == sensor_b:
                matrix.loc[sensor_a, sensor_b] = 0.0
                continue
            xdiv = compute_cross_sensor_divergence(
                models[sensor_a].accx_reference_distributions,
                models[sensor_b].accx_reference_distributions,
            )
            matrix.loc[sensor_a, sensor_b] = xdiv.get('xdiv_mean_kl', np.nan)
            for key, value in xdiv.items():
                if key == 'xdiv_mean_kl':
                    continue
                band_rows.append({
                    'sensor_a': sensor_a,
                    'sensor_b': sensor_b,
                    'band': key.replace('xdiv_', '').replace('_kl', '').upper(),
                    'kl': value,
                })

    return matrix, pd.DataFrame(band_rows)


def run_fault_validation(
    trimmed_signals: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    models: Mapping[str, SensorHealthModel],
    fs: float = DEFAULT_FS,
    fault_duration_sec: int = FAULT_BASE_DURATION_SEC,
    severities: Sequence[float] = tuple(np.linspace(0.0, 2.0, 7)),
    n_trials: int = 3,
    fault_types: Sequence[str] = FAULT_TYPES,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    wavelet_level: int = DEFAULT_WAVELET_LEVEL,
    seed: int = 42,
) -> pd.DataFrame:
    base_len = int(fault_duration_sec * fs)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float | str | int]] = []

    for sensor, sensor_model in models.items():
        print(f'[faults] sensor={sensor}')
        base_axes = build_fault_reference_axes(
            trimmed_signals[sensor],
            fs=fs,
            duration_sec=fault_duration_sec,
        )
        axis_names = list(DEFAULT_AXES.keys())
        stacked = np.column_stack([base_axes[axis][:base_len] for axis in axis_names])

        for fault_type in fault_types:
            print(f'[faults]   mode={fault_type}')
            for severity in severities:
                for trial in range(n_trials):
                    local_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
                    degraded = inject_synthetic_fault(stacked, fault_type, severity=float(severity), rng=local_rng)
                    degraded_axes = {
                        axis_name: degraded[:, idx]
                        for idx, axis_name in enumerate(axis_names)
                    }

                    feat_df = _extract_multiaxis_feature_df(
                        degraded_axes,
                        fs=fs,
                        window_sec=window_sec,
                        step_sec=step_sec,
                        wavelet_level=wavelet_level,
                    )
                    qv = build_quality_vector(
                        feat_df,
                        sensor_model.quality_features,
                        normalize=True,
                        reference_stats=sensor_model.reference_stats,
                    )
                    qv = apply_health_score_calibration(qv, sensor_model.health_calibration)

                    div_summary = summarize_signal_divergence(
                        degraded_axes['AccX'],
                        ref_dists=sensor_model.accx_reference_distributions,
                        fs=fs,
                        window_sec=window_sec,
                        step_sec=step_sec,
                        level=wavelet_level,
                    )

                    rows.append({
                        'sensor': sensor,
                        'fault_type': fault_type,
                        'severity': float(severity),
                        'trial': int(trial),
                        'mean_health_score': float(qv['qv_health_score'].mean()),
                        'mean_health_score_raw': float(qv['qv_health_score_raw'].mean()),
                        'mean_div_mean_kl': div_summary['mean_div_mean_kl'],
                        'mean_div_mean_jsd': div_summary['mean_div_mean_jsd'],
                    })

    return pd.DataFrame(rows)


def summarize_condition_metrics(
    models: Mapping[str, SensorHealthModel],
) -> pd.DataFrame:
    rows: List[Dict[str, float | str]] = []

    for sensor, model in models.items():
        divergence_grouped = model.divergence_timeline.groupby('condition', as_index=False).agg({
            'div_mean_kl': 'mean',
            'div_mean_jsd': 'mean',
        })
        divergence_lookup = {
            row['condition']: row
            for _, row in divergence_grouped.iterrows()
        }

        for condition, qv in model.condition_qv.items():
            rows.append({
                'sensor': sensor,
                'condition': condition,
                'condition_label': CONDITION_LABELS[condition],
                'rpm': RPM_VALUES[condition],
                'mean_health_score': float(qv['qv_health_score'].mean()),
                'std_health_score': float(qv['qv_health_score'].std()),
                'mean_health_score_raw': float(qv['qv_health_score_raw'].mean()),
                'mean_div_mean_kl': float(divergence_lookup[condition]['div_mean_kl']),
                'mean_div_mean_jsd': float(divergence_lookup[condition]['div_mean_jsd']),
            })

    return pd.DataFrame(rows)


def run_shield_health_analysis(
    data_root: Path | str,
    fs: float = DEFAULT_FS,
    window_sec: float = DEFAULT_WINDOW_SEC,
    step_sec: float = DEFAULT_STEP_SEC,
    wavelet_level: int = DEFAULT_WAVELET_LEVEL,
    healthy_quantile: float = HEALTH_QUANTILE,
    seed: int = 42,
) -> Dict[str, object]:
    print('[pipeline] loading recollection data')
    data_by_sensor = load_shield_recollection(data_root)
    print('[pipeline] trimming calibrated signals')
    trimmed_signals = build_trimmed_signals(data_by_sensor, fs=fs)
    print('[pipeline] extracting multiaxis features')
    per_condition_features = build_sensor_feature_maps(
        trimmed_signals,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        wavelet_level=wavelet_level,
    )

    print('[pipeline] fitting per-sensor healthy models')
    models = {
        sensor: build_sensor_health_model(
            sensor,
            per_condition_features[sensor],
            trimmed_signals[sensor],
            fs=fs,
            healthy_quantile=healthy_quantile,
            window_sec=window_sec,
            step_sec=step_sec,
            wavelet_level=wavelet_level,
        )
        for sensor in per_condition_features
    }

    print('[pipeline] computing cross-sensor divergence')
    cross_sensor_matrix, cross_sensor_bands = build_cross_sensor_divergence(models)
    print('[pipeline] running synthetic fault validation')
    fault_results = run_fault_validation(
        trimmed_signals,
        models,
        fs=fs,
        window_sec=window_sec,
        step_sec=step_sec,
        wavelet_level=wavelet_level,
        seed=seed,
    )

    return {
        'data_by_sensor': data_by_sensor,
        'trimmed_signals': trimmed_signals,
        'modwt_segments': build_modwt_segments(trimmed_signals, fs=fs),
        'sensor_models': models,
        'condition_summary': summarize_condition_metrics(models),
        'cross_sensor_matrix': cross_sensor_matrix,
        'cross_sensor_bands': cross_sensor_bands,
        'fault_results': fault_results,
        'sanity_summary': build_stationary_sanity_summary(data_by_sensor),
    }
