import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from signal_pattern_classification.statistical_features import (
    streaming_windows,
    stored_windows,
    extract_features_from_windows,
    top_k_accuracy,
    FEATURE_NAMES
)

SHL_COLUMN_MAP = {
    'accelerometer': [1, 2, 3],
    'gyroscope': [4, 5, 6],
    'magnetometer': [7, 8, 9],
}

WINDOW_SIZES = [10, 20, 50, 100]
MODES = ['streaming', 'stored']


def load_shl_motion_file(filepath: str, max_rows: int = None) -> dict:
    print(f"Loading {os.path.basename(filepath)}...")
    data = np.loadtxt(filepath, max_rows=max_rows)

    nan_mask = ~np.isnan(data).any(axis=1)
    data = data[nan_mask]

    modality_signals = {}
    for modality, cols in SHL_COLUMN_MAP.items():
        signals = []
        for col in cols:
            signal = data[:, col]
            if len(signal) >= 100:
                signals.append(signal)
        modality_signals[modality] = signals

    return modality_signals


def load_multiple_shl_files(data_dir: str, max_rows_per_file: int = 100000) -> dict:
    all_signals = {m: [] for m in SHL_COLUMN_MAP.keys()}

    for session in ['220617', '260617', '270617']:
        session_dir = os.path.join(data_dir, session)
        if not os.path.exists(session_dir):
            continue

        for position in ['Hand', 'Bag', 'Hips', 'Torso']:
            filepath = os.path.join(session_dir, f'{position}_Motion.txt')
            if not os.path.exists(filepath):
                continue

            signals = load_shl_motion_file(filepath, max_rows=max_rows_per_file)
            for modality, sigs in signals.items():
                all_signals[modality].extend(sigs)

    return all_signals


def create_data_splits(modality_data: dict, random_state: int = 42) -> tuple:
    rng = np.random.default_rng(random_state)

    total_samples = {m: sum(len(s) for s in signals) for m, signals in modality_data.items()}
    min_samples = min(total_samples.values())
    split_size = int(0.2 * min_samples)

    print(f"\nSamples per modality: {total_samples}")
    print(f"Smallest class: {min_samples}, split size: {split_size}")

    train_data = {m: [] for m in SHL_COLUMN_MAP.keys()}
    eval_data = {m: [] for m in SHL_COLUMN_MAP.keys()}

    for modality, signals in modality_data.items():
        all_samples = np.concatenate(signals)
        rng.shuffle(all_samples)

        eval_data[modality] = [all_samples[:split_size]]
        train_data[modality] = [all_samples[split_size:]]

    return train_data, eval_data


def prepare_dataset(data: dict, window_size: int, mode: str, random_state: int = 42) -> tuple:
    X_list = []
    y_list = []

    for modality, signals in data.items():
        for signal in signals:
            if mode == 'streaming':
                windows = streaming_windows(signal, window_size)
            else:
                n_windows = len(signal) // window_size
                if n_windows == 0:
                    continue
                windows = stored_windows(signal, window_size, n_windows, random_state)

            if len(windows) == 0:
                continue

            features = extract_features_from_windows(windows)
            X_list.append(features)
            y_list.extend([modality] * len(features))

    if not X_list:
        return np.array([]).reshape(0, 5), np.array([])

    return np.vstack(X_list), np.array(y_list)


def train_classifier(train_data: dict, window_size: int) -> RandomForestClassifier:
    X_train, y_train = prepare_dataset(train_data, window_size, 'streaming')
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def run_experiment(clf: RandomForestClassifier, eval_data: dict, window_size: int, mode: str) -> dict:
    X_test, y_test = prepare_dataset(eval_data, window_size, mode)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    top1 = top_k_accuracy(y_test, y_proba, k=1, classes=clf.classes_)
    top2 = top_k_accuracy(y_test, y_proba, k=2, classes=clf.classes_)
    top3 = top_k_accuracy(y_test, y_proba, k=3, classes=clf.classes_)

    return {'top1': top1, 'top2': top2, 'top3': top3}


def run_all_experiments(train_data: dict, eval_data: dict) -> dict:
    results = {}
    classifiers = {}

    for ws in WINDOW_SIZES:
        print(f"\nTraining classifier for window_size={ws}...")
        classifiers[ws] = train_classifier(train_data, ws)

    print("\nRunning experiments...")
    for ws in WINDOW_SIZES:
        for mode in MODES:
            key = (ws, mode)
            results[key] = run_experiment(classifiers[ws], eval_data, ws, mode)
            print(f"  Window={ws:3d}, Mode={mode:9s}: "
                  f"Top-1={results[key]['top1']*100:5.1f}%, "
                  f"Top-2={results[key]['top2']*100:5.1f}%, "
                  f"Top-3={results[key]['top3']*100:5.1f}%")

    return results


def generate_results_table(results: dict) -> pd.DataFrame:
    rows = []
    for ws in WINDOW_SIZES:
        for mode in MODES:
            key = (ws, mode)
            r = results[key]
            rows.append({
                'Window Size': ws,
                'Mode': mode,
                'Top-1 (%)': f"{r['top1']*100:.1f}",
                'Top-2 (%)': f"{r['top2']*100:.1f}",
                'Top-3 (%)': f"{r['top3']*100:.1f}"
            })
    return pd.DataFrame(rows)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', '..', 'SHL Dataset', 'SHLDataset_preview_v2', 'User1')

    print("=" * 60)
    print("SHL Dataset: Signal Pattern Classifier Test")
    print("=" * 60)

    print("\n[1/3] Loading SHL data (sampling subset for speed)...")
    modality_data = load_multiple_shl_files(data_dir, max_rows_per_file=50000)

    for m, signals in modality_data.items():
        total = sum(len(s) for s in signals)
        print(f"  {m}: {len(signals)} signals, {total:,} samples")

    print("\n[2/3] Creating train/eval splits...")
    train_data, eval_data = create_data_splits(modality_data)

    print("\n[3/3] Training and evaluating...")
    results = run_all_experiments(train_data, eval_data)

    print("\n" + "=" * 60)
    print("RESULTS TABLE - SHL Dataset")
    print("=" * 60)
    results_df = generate_results_table(results)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"PAMAP2 Dataset (4 modalities): ~100% Top-1")
    print(f"SHL Dataset (3 modalities):    {results[(20, 'streaming')]['top1']*100:.1f}% Top-1")


if __name__ == '__main__':
    main()
