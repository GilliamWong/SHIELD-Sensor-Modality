import numpy as np
import pandas as pd

from physics_based_classification.fault_injection import FAULT_TYPES, inject_synthetic_fault
from physics_based_classification.quality_vector import (
    apply_health_score_calibration,
    build_quality_vector,
    fit_health_score_calibration,
)
from physics_based_classification.shield_health_pipeline import load_shield_recollection


def test_health_score_calibration_bounds_and_envelope():
    raw_scores = pd.Series([0.10, 0.12, 0.14, 0.18, 0.20, 0.90])
    calibration = fit_health_score_calibration(raw_scores, healthy_quantile=0.8)
    qv = pd.DataFrame({'qv_health_score_raw': raw_scores})

    result = apply_health_score_calibration(qv, calibration)

    assert result['qv_health_score'].between(0.0, 1.0).all()
    threshold = calibration['healthy_threshold']
    in_envelope = result['qv_health_score_raw'] <= threshold + 1e-12
    assert (result.loc[in_envelope, 'qv_health_score'] == 1.0).all()
    assert result.loc[result['qv_health_score_raw'].idxmax(), 'qv_health_score'] < 1.0


def test_build_quality_vector_preserves_raw_health_distance():
    df = pd.DataFrame({
        'feat_a': [0.0, 1.0, 2.0],
        'feat_b': [1.0, 2.0, 3.0],
        'window_start_sec': [0.0, 1.0, 2.0],
    })
    qf = {
        'all': ['feat_a', 'feat_b'],
        'stable': ['feat_a'],
        'sensitive': ['feat_b'],
        'mandatory': ['feat_b'],
    }

    result = build_quality_vector(df, qf, normalize=False)

    assert 'qv_health_score_raw' in result.columns
    assert 'qv_health_score' in result.columns
    assert 'qv_degradation_score_raw' in result.columns
    assert np.allclose(result['qv_health_score_raw'], result['qv_health_score'])


def test_inject_synthetic_fault_preserves_shape_for_all_modes():
    rng = np.random.default_rng(42)
    signal = rng.normal(size=(256, 4))

    for fault_type in FAULT_TYPES:
        degraded = inject_synthetic_fault(signal, fault_type, severity=1.0, rng=rng)
        assert degraded.shape == signal.shape


def test_load_shield_recollection_reads_calibrated_schema(tmp_path):
    rows = [
        {
            'Time_ms': 0,
            'CalAX': 0.1, 'CalAY': 0.2, 'CalAZ': 9.8,
            'CalGX': 0.0, 'CalGY': 0.0, 'CalGZ': 0.0,
            'RawAX': 1.0, 'RawAY': 2.0, 'RawAZ': 3.0,
            'RawGX': 4.0, 'RawGY': 5.0, 'RawGZ': 6.0,
        },
        {
            'Time_ms': 10,
            'CalAX': 0.1, 'CalAY': 0.2, 'CalAZ': 9.8,
            'CalGX': 0.0, 'CalGY': 0.0, 'CalGZ': 0.0,
            'RawAX': 1.0, 'RawAY': 2.0, 'RawAZ': 3.0,
            'RawGX': 4.0, 'RawGY': 5.0, 'RawGZ': 6.0,
        },
    ]
    df = pd.DataFrame(rows)
    filenames = [
        'dfrobot_stationary_test_20260302-212758.csv',
        'dfrobot_40_rpm_calibrated_20260302-223145.csv',
        'dfrobot_80_rpm_calibrated_20260302-224940.csv',
        'dfrobot_120_rpm_calibrated_20260302-231218.csv',
        'dfrobot_160_rpm_calibrated_20260302-233242.csv',
        'dfrobot_200_rpm_calibrated_20260302-235416.csv',
        'pololu_stationary_test_calibrated_20260303-002140.csv',
        'pololu_40_rpm_calibrated_20260303-012500.csv',
        'pololu_80_rpm_calibrated_20260303-014244.csv',
        'pololu_120_rpm_calibrated_20260303-020055.csv',
        'pololu_160_rpm_calibrated_20260303-021855.csv',
        'pololu_200_rpm_calibrated_20260303-024724.csv',
    ]
    for name in filenames:
        df.to_csv(tmp_path / name, index=False)

    loaded = load_shield_recollection(tmp_path)

    assert set(loaded.keys()) == {'DFRobot', 'Pololu'}
    assert set(loaded['DFRobot'].keys()) == {
        'stationary', '40_rpm', '80_rpm', '120_rpm', '160_rpm', '200_rpm'
    }
    assert 'CalAX' in loaded['Pololu']['stationary'].columns
