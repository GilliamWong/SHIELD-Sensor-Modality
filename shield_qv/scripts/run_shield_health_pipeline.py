#!/usr/bin/env python3
"""
Run the SHIELD healthy-envelope validation pipeline on the March 2 recollection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PACKAGE_ROOT / 'src'))

from physics_based_classification.shield_health_pipeline import (
    CONDITION_LABELS,
    CONDITIONS,
    DEFAULT_FS,
    RPM_VALUES,
    run_shield_health_analysis,
)
from physics_based_classification.wavelet_analyses import modwt_fast


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_sanity_summary(sanity_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x = np.arange(len(sanity_df))
    width = 0.35

    axes[0].bar(x - width / 2, sanity_df['raw_acc_norm_mean'], width, label='Raw', color='#c44e52')
    axes[0].bar(x + width / 2, sanity_df['cal_acc_norm_mean'], width, label='Calibrated', color='#4c72b0')
    axes[0].axhline(9.81, color='black', linestyle='--', linewidth=1, alpha=0.6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sanity_df['sensor'])
    axes[0].set_ylabel('Mean |Accel|')
    axes[0].set_title('Stationary Accel Norm')
    axes[0].legend()

    axes[1].bar(x - width / 2, sanity_df['raw_gyro_norm_mean'], width, label='Raw', color='#c44e52')
    axes[1].bar(x + width / 2, sanity_df['cal_gyro_norm_mean'], width, label='Calibrated', color='#4c72b0')
    axes[1].axhline(0.0, color='black', linestyle='--', linewidth=1, alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sanity_df['sensor'])
    axes[1].set_ylabel('Mean |Gyro|')
    axes[1].set_title('Stationary Gyro Norm')
    axes[1].legend()

    fig.suptitle('March 2 Recollection: Raw vs Calibrated Sanity Check')
    plt.tight_layout()
    _savefig(output_dir / 'raw_vs_calibrated_sanity.png')


def plot_modwt_decomposition(segments: dict[str, np.ndarray], fs: float, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

    for ax, (sensor, segment) in zip(axes, segments.items()):
        decomp = modwt_fast(segment, level=5)
        t = np.arange(len(segment)) / fs
        ax.plot(t, segment, color='black', linewidth=0.7, label='AccX')
        ax.plot(t, decomp[f'A{decomp["levels"]}'], color='#4c72b0', linewidth=1.0, label=f'A{decomp["levels"]}')
        ax.set_title(f'{sensor} Stationary AccX (Calibrated) with MODWT Approximation')
        ax.set_ylabel('Signal')
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    _savefig(output_dir / 'modwt_decomposition_stationary.png')


def plot_healthy_condition_summary(condition_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = list(CONDITIONS)
    x = np.arange(len(order))
    width = 0.35

    for idx, sensor in enumerate(sorted(condition_df['sensor'].unique())):
        sub = condition_df[condition_df['sensor'] == sensor].set_index('condition').loc[order].reset_index()
        offset = -width / 2 if idx == 0 else width / 2
        axes[0].bar(x + offset, sub['mean_health_score'], width, yerr=sub['std_health_score'],
                    label=sensor, alpha=0.85)
        axes[1].plot(x, sub['mean_div_mean_kl'], marker='o', linewidth=2, label=f'{sensor} KL')
        axes[1].plot(x, sub['mean_div_mean_jsd'], marker='s', linewidth=2, linestyle='--', label=f'{sensor} JSD')

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([CONDITION_LABELS[c] for c in order], rotation=45, ha='right')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel('Mean Health Score')
    axes[0].set_title('Part 1: Healthy Conditions Stay Healthy')
    axes[0].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CONDITION_LABELS[c] for c in order], rotation=45, ha='right')
    axes[1].set_ylabel('Mean Divergence')
    axes[1].set_title('Part 1: Healthy KL / JSD by Condition')
    axes[1].legend(fontsize=8, ncol=2)

    plt.tight_layout()
    _savefig(output_dir / 'healthy_condition_summary.png')


def plot_timeline(models: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, sensor in zip(axes, sorted(models)):
        timeline = models[sensor].timeline_qv
        ax.plot(timeline['window_start_sec_global'] / 60.0, timeline['qv_health_score'],
                linewidth=0.8, color='#4c72b0')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Health')
        ax.set_title(f'{sensor}: 0-1 Health Score Across Healthy Conditions')

        for condition in CONDITIONS[1:]:
            boundary = timeline[timeline['condition'] == condition]['window_start_sec_global'].min() / 60.0
            ax.axvline(boundary, color='gray', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    _savefig(output_dir / 'healthy_health_timeline.png')


def plot_divergence_timeline(models: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    for ax, sensor in zip(axes, sorted(models)):
        timeline = models[sensor].divergence_timeline
        ax.plot(timeline['window_start_sec_global'] / 60.0, timeline['div_mean_kl'],
                linewidth=0.8, color='#dd8452', label='Mean KL')
        ax.plot(timeline['window_start_sec_global'] / 60.0, timeline['div_mean_jsd'],
                linewidth=0.8, color='#55a868', label='Mean JSD')
        ax.set_ylabel('Divergence')
        ax.set_title(f'{sensor}: AccX Divergence vs Pooled Healthy Reference')
        ax.legend()

        for condition in CONDITIONS[1:]:
            boundary = timeline[timeline['condition'] == condition]['window_start_sec_global'].min() / 60.0
            ax.axvline(boundary, color='gray', linestyle=':', alpha=0.4)

    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    _savefig(output_dir / 'healthy_divergence_timeline.png')


def plot_cross_sensor_divergence(matrix_df: pd.DataFrame, band_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].imshow(matrix_df.to_numpy(dtype=float), cmap='magma_r')
    axes[0].set_xticks(range(len(matrix_df.columns)))
    axes[0].set_xticklabels(matrix_df.columns)
    axes[0].set_yticks(range(len(matrix_df.index)))
    axes[0].set_yticklabels(matrix_df.index)
    axes[0].set_title('Cross-Sensor Mean KL')
    for i in range(len(matrix_df.index)):
        for j in range(len(matrix_df.columns)):
            axes[0].text(j, i, f'{matrix_df.iloc[i, j]:.3f}', ha='center', va='center', color='white', fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    if band_df.empty:
        axes[1].text(0.5, 0.5, 'No band-level KL available', ha='center', va='center')
        axes[1].axis('off')
    else:
        sub = band_df[(band_df['sensor_a'] == 'DFRobot') & (band_df['sensor_b'] == 'Pololu')]
        sub = sub.sort_values('band')
        axes[1].bar(sub['band'], sub['kl'], color='#8172b3')
        axes[1].set_ylabel('KL')
        axes[1].set_title('DFRobot vs Pololu Band KL')
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    _savefig(output_dir / 'cross_sensor_divergence.png')


def plot_fault_response(fault_df: pd.DataFrame, output_dir: Path) -> None:
    fault_types = list(dict.fromkeys(fault_df['fault_type']))
    ncols = 3
    nrows = int(np.ceil(len(fault_types) / ncols))

    fig_h, axes_h = plt.subplots(nrows, ncols, figsize=(15, 8), sharex=True, sharey=True)
    axes_h = np.array(axes_h).reshape(-1)
    fig_d, axes_d = plt.subplots(nrows, ncols, figsize=(15, 8), sharex=True, sharey=True)
    axes_d = np.array(axes_d).reshape(-1)

    for ax_h, ax_d, fault_type in zip(axes_h, axes_d, fault_types):
        sub = fault_df[fault_df['fault_type'] == fault_type]
        grouped = sub.groupby(['sensor', 'severity'], as_index=False).agg({
            'mean_health_score': ['mean', 'std'],
            'mean_div_mean_kl': ['mean', 'std'],
        })
        grouped.columns = ['sensor', 'severity', 'health_mean', 'health_std', 'kl_mean', 'kl_std']

        for sensor in sorted(grouped['sensor'].unique()):
            sensor_df = grouped[grouped['sensor'] == sensor].sort_values('severity')
            ax_h.plot(sensor_df['severity'], sensor_df['health_mean'], marker='o', linewidth=2, label=sensor)
            ax_h.fill_between(
                sensor_df['severity'],
                np.clip(sensor_df['health_mean'] - sensor_df['health_std'], 0.0, 1.0),
                np.clip(sensor_df['health_mean'] + sensor_df['health_std'], 0.0, 1.0),
                alpha=0.15,
            )

            ax_d.plot(sensor_df['severity'], sensor_df['kl_mean'], marker='o', linewidth=2, label=sensor)
            ax_d.fill_between(
                sensor_df['severity'],
                np.clip(sensor_df['kl_mean'] - sensor_df['kl_std'], 0.0, None),
                sensor_df['kl_mean'] + sensor_df['kl_std'],
                alpha=0.15,
            )

        ax_h.set_title(fault_type.replace('_', ' ').title())
        ax_h.set_ylim(0, 1.05)
        ax_h.set_xlabel('Severity')
        ax_h.set_ylabel('Health')

        ax_d.set_title(fault_type.replace('_', ' ').title())
        ax_d.set_xlabel('Severity')
        ax_d.set_ylabel('Mean KL')

    for ax in axes_h[len(fault_types):]:
        ax.axis('off')
    for ax in axes_d[len(fault_types):]:
        ax.axis('off')

    axes_h[0].legend()
    axes_d[0].legend()
    fig_h.suptitle('Part 2: Health Score Drops Under Synthetic Fault Injection')
    fig_d.suptitle('Part 2: KL Divergence Rises Under Synthetic Fault Injection')
    fig_h.tight_layout()
    fig_d.tight_layout()
    _savefig(output_dir / 'fault_health_response.png')
    _savefig(output_dir / 'fault_divergence_response.png')


def write_summary_files(results: dict, output_dir: Path) -> None:
    sensor_models = results['sensor_models']
    summary = {
        'data_root': str(results['data_root']),
        'selected_features': {
            sensor: model.quality_features
            for sensor, model in sensor_models.items()
        },
        'health_calibration': {
            sensor: model.health_calibration
            for sensor, model in sensor_models.items()
        },
        'condition_health': results['condition_summary'].to_dict(orient='records'),
    }
    with (output_dir / 'summary.json').open('w') as f:
        json.dump(summary, f, indent=2)

    results['sanity_summary'].to_csv(output_dir / 'sanity_summary.csv', index=False)
    results['condition_summary'].to_csv(output_dir / 'condition_summary.csv', index=False)
    results['fault_results'].to_csv(output_dir / 'fault_results.csv', index=False)
    results['cross_sensor_matrix'].to_csv(output_dir / 'cross_sensor_matrix.csv')
    results['cross_sensor_bands'].to_csv(output_dir / 'cross_sensor_bands.csv', index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-root',
        type=Path,
        default=REPO_ROOT / 'datasets/SHIELD Data Collection/march 2 data (recollection of pololu and dfrobot)',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=PACKAGE_ROOT / 'notebooks/figures/shield_health_validation',
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fs', type=float, default=DEFAULT_FS)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_shield_health_analysis(args.data_root, fs=args.fs, seed=args.seed)
    results['data_root'] = args.data_root

    write_summary_files(results, output_dir)
    plot_sanity_summary(results['sanity_summary'], output_dir)
    plot_modwt_decomposition(results['modwt_segments'], args.fs, output_dir)
    plot_healthy_condition_summary(results['condition_summary'], output_dir)
    plot_timeline(results['sensor_models'], output_dir)
    plot_divergence_timeline(results['sensor_models'], output_dir)
    plot_cross_sensor_divergence(results['cross_sensor_matrix'], results['cross_sensor_bands'], output_dir)
    plot_fault_response(results['fault_results'], output_dir)

    print(f'Wrote SHIELD health validation outputs to {output_dir}')


if __name__ == '__main__':
    main()
