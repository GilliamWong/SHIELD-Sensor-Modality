import numpy as np
import pandas as pd
import glob
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SensorDataLoader import SensorDataLoader
from signal_pattern_classification.statistical_features import (
    streaming_windows,
    stored_windows,
    extract_features_from_windows,
    top_k_accuracy,
    FEATURE_NAMES
)

MODALITY_MAP = {
    'accelerometer': ['hand_accel_16g', 'chest_accel_16g', 'ankle_accel_16g'],
    'gyroscope': ['hand_gyro', 'chest_gyro', 'ankle_gyro'],
    'magnetometer': ['hand_mag', 'chest_mag', 'ankle_mag'],
    'temperature': ['hand_temp', 'chest_temp', 'ankle_temp']
}

WINDOW_SIZES = [10, 20, 50, 100]
MODES = ['streaming', 'stored']


def load_all_pamap2_data(data_dir: str) -> dict:
    loader = SensorDataLoader(seed=42)
    files = sorted(glob.glob(os.path.join(data_dir, 'subject*.dat')))

    modality_signals = {m: [] for m in MODALITY_MAP.keys()}

    for filepath in files:
        print(f"Loading {os.path.basename(filepath)}...")
        sensors = loader.load_pamap2(filepath)
        sensors = loader.get_stationary_segments(sensors, activities=[2, 3])

        for modality, sensor_keys in MODALITY_MAP.items():
            for key in sensor_keys:
                if key not in sensors:
                    continue
                data = sensors[key]
                if data.ndim == 1:
                    clean = data[~np.isnan(data)]
                    if len(clean) >= 100:
                        modality_signals[modality].append(clean)
                else:
                    for axis in range(data.shape[1]):
                        clean = data[:, axis]
                        clean = clean[~np.isnan(clean)]
                        if len(clean) >= 100:
                            modality_signals[modality].append(clean)

    return modality_signals


def create_data_splits(modality_data: dict, random_state: int = 42) -> tuple:
    rng = np.random.default_rng(random_state)

    total_samples = {m: sum(len(s) for s in signals) for m, signals in modality_data.items()}
    min_samples = min(total_samples.values())
    split_size = int(0.2 * min_samples)

    print(f"\nSamples per modality: {total_samples}")
    print(f"Smallest class: {min_samples}, split size: {split_size}")

    train_data = {m: [] for m in MODALITY_MAP.keys()}
    val_data = {m: [] for m in MODALITY_MAP.keys()}
    eval_data = {m: [] for m in MODALITY_MAP.keys()}

    for modality, signals in modality_data.items():
        all_samples = np.concatenate(signals)
        rng.shuffle(all_samples)

        eval_data[modality] = [all_samples[:split_size]]
        val_data[modality] = [all_samples[split_size:2*split_size]]
        train_data[modality] = [all_samples[2*split_size:]]

    return train_data, val_data, eval_data


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


def run_experiment(clf: RandomForestClassifier, eval_data: dict, window_size: int, mode: str, random_state: int = 42) -> dict:
    X_test, y_test = prepare_dataset(eval_data, window_size, mode, random_state)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    top1 = top_k_accuracy(y_test, y_proba, k=1, classes=clf.classes_)
    top2 = top_k_accuracy(y_test, y_proba, k=2, classes=clf.classes_)
    top3 = top_k_accuracy(y_test, y_proba, k=3, classes=clf.classes_)

    return {
        'top1': top1,
        'top2': top2,
        'top3': top3,
        'y_true': y_test,
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=clf.classes_),
        'classes': clf.classes_
    }


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
    data_dir = os.path.join(script_dir, '..', '..', 'PAMAP2_Dataset', 'Protocol')

    print("="*60)
    print("Paper-Style Sensor Modality Classifier")
    print("Based on Culic Gambiroza et al. (2025)")
    print("="*60)

    print("\n[1/4] Loading PAMAP2 data...")
    modality_data = load_all_pamap2_data(data_dir)

    for m, signals in modality_data.items():
        total = sum(len(s) for s in signals)
        print(f"  {m}: {len(signals)} signals, {total:,} samples")

    print("\n[2/4] Creating train/val/eval splits...")
    train_data, val_data, eval_data = create_data_splits(modality_data)

    print("\n[3/4] Training and evaluating...")
    results = run_all_experiments(train_data, eval_data)

    print("\n" + "="*60)
    print("RESULTS TABLE")
    print("="*60)
    results_df = generate_results_table(results)
    print(results_df.to_string(index=False))

    print("\n" + "="*60)
    print("CONFUSION MATRIX (Window=20, Streaming)")
    print("="*60)
    key = (20, 'streaming')
    cm = results[key]['confusion_matrix']
    classes = results[key]['classes']
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df)

    print("\n" + "="*60)
    print("COMPARISON TO SHIELD BASELINE")
    print("="*60)
    print(f"SHIELD Physics-Informed (2.0s window, ~15 features): 99.2% Top-1")
    print(f"Paper Style (20 samples, 5 features, streaming):     {results[(20, 'streaming')]['top1']*100:.1f}% Top-1")
    print(f"Paper Style (20 samples, 5 features, streaming):     {results[(20, 'streaming')]['top2']*100:.1f}% Top-2")


if __name__ == '__main__':
    main()
