# SHIELD Quality Vector Validation

## Canonical Package Workflow

The `shield_qv` package now contains the validated March 2 SHIELD health-monitoring workflow:

1. Load the March 2 recollection data for DFRobot and Pololu from the repo-level `datasets/SHIELD Data Collection/march 2 data (recollection of pololu and dfrobot)` folder.
2. Use calibrated channels (`CalAX/CalAY/CalAZ/CalGX/CalGY/CalGZ`) as the analysis source of truth.
3. Fit feature references and a bounded 0–1 health score using healthy data only, where `1.0 = healthy`.
4. Validate that healthy shaking across the benign RPM conditions stays healthy.
5. Inject synthetic faults into healthy motion-rich windows and verify that health decreases and divergence rises.

This package-local implementation is driven by:

- [run_shield_health_pipeline.py](/Users/gilliam/Desktop/Project%20Shield/shield_qv/scripts/run_shield_health_pipeline.py)
- [shield_health_pipeline.py](/Users/gilliam/Desktop/Project%20Shield/shield_qv/src/physics_based_classification/shield_health_pipeline.py)
- [quality_vector.py](/Users/gilliam/Desktop/Project%20Shield/shield_qv/src/physics_based_classification/quality_vector.py)
- [fault_injection.py](/Users/gilliam/Desktop/Project%20Shield/shield_qv/src/physics_based_classification/fault_injection.py)

## Scoring Semantics

- `qv_health_score_raw` is the unbounded anomaly distance in normalized feature space.
- `qv_health_score` is the calibrated public score in `[0, 1]`.
- `1.0` means the window remains inside the healthy envelope learned from healthy data.
- Values below `1.0` indicate excess deviation beyond that healthy envelope.

## Experimental Narrative

### Part 1: Healthy-condition validation

- Calibrate with healthy calibrated data across all benign operating conditions.
- Extract multiaxis features and pooled-healthy KL/JSD divergence.
- Confirm that the health score remains high during healthy shaking.
- Conclusion: the experimental procedure is stable under benign operating changes.

### Part 2: Synthetic-fault validation

- Reuse the healthy calibration.
- Inject synthetic faults into healthy motion-rich reference windows.
- Confirm that the health score drops and divergence rises with severity.
- Conclusion: the quality-vector pipeline is sensitive to sensor faults rather than normal operating motion alone.

Real fault data remains the next extension after this synthetic-fault validation step. Bosch is still out of scope in this package snapshot because it uses a separate collection stack.

## Outputs

The canonical executed report notebook is:

- [shield_health_validation.ipynb](/Users/gilliam/Desktop/Project%20Shield/shield_qv/notebooks/shield_health_validation.ipynb)

The regenerated figures and summary tables are stored in:

- [shield_health_validation](/Users/gilliam/Desktop/Project%20Shield/shield_qv/notebooks/figures/shield_health_validation)

Key artifacts include:

- `healthy_condition_summary.png`
- `healthy_health_timeline.png`
- `healthy_divergence_timeline.png`
- `fault_health_response.png`
- `fault_divergence_response.png`
- `condition_summary.csv`
- `fault_results.csv`
- `summary.json`

## Expected Results

With the current March 2 calibrated data and bounded health-score calibration:

- Healthy DFRobot and Pololu windows remain near `1.0` across benign RPM conditions.
- Synthetic `noise_increase`, `bias_drift`, `spike_injection`, `bandwidth_loss`, `saturation`, and `gain_change` faults all reduce the mean health score at high severity.
- KL divergence rises under injected faults even though healthy shaking remains comparatively stable, which matches the intended interpretation that the method tracks changes in the sensor-state distribution rather than raw motion magnitude.
