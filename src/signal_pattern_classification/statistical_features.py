import numpy as np

def extract_window_features(window: np.ndarray) -> dict:
    return {
        'min': float(np.min(window)),
        'max': float(np.max(window)),
        'avg': float(np.mean(window)),
        'median': float(np.median(window)),
        'std': float(np.std(window))
    }

def streaming_windows(signal: np.ndarray, window_size: int) -> np.ndarray:
    n_windows = len(signal) // window_size
    if n_windows == 0:
        return np.array([]).reshape(0, window_size)
    return signal[:n_windows * window_size].reshape(n_windows, window_size)

def stored_windows(signal: np.ndarray, window_size: int, n_windows: int, random_state: int = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    max_start = len(signal) - window_size
    if max_start < 0:
        return np.array([]).reshape(0, window_size)
    starts = rng.integers(0, max_start + 1, size=n_windows)
    return np.array([signal[s:s + window_size] for s in starts])

def extract_features_from_windows(windows: np.ndarray) -> np.ndarray:
    if len(windows) == 0:
        return np.array([]).reshape(0, 5)
    return np.column_stack([
        np.min(windows, axis=1),
        np.max(windows, axis=1),
        np.mean(windows, axis=1),
        np.median(windows, axis=1),
        np.std(windows, axis=1)
    ])

def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int, classes: np.ndarray) -> float:
    top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_idx = np.array([class_to_idx[y] for y in y_true])
    correct = np.array([y in preds for y, preds in zip(y_true_idx, top_k_indices)])
    return float(np.mean(correct))

FEATURE_NAMES = ['min', 'max', 'avg', 'median', 'std']
