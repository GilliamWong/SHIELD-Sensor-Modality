import numpy as np
import pandas as pd
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
from .wavelet_analyses import extract_wavelet_features
from .signal_quality import extract_signal_quality_features
from .divergence_analysis import compute_window_divergence

from . import freq_domain_analyses
from . import time_domain_analyses
from . import allan_dev


class FeatureExtractor:
    def __init__(self, fs: float):
        self.fs = fs

    def _extract_features(
        self,
        window: np.ndarray,
        bands: Optional[Iterable[Sequence[float]]] = None,
        include_adev: bool = True,
        adev_min_samples: int = 50,
        include_wavelet: bool = True,
        wavelet_level: Optional[int] = 5,
        include_signal_quality: bool = True,
        include_divergence: bool = False,
        divergence_ref_dists: Optional[Dict[str, Tuple]] = None,
    ) -> dict:
        """Extract all feature categories from a single segment."""
        n = len(window)

        # Time domain
        stats = time_domain_analyses.get_statistical_moments(window)

        # Frequency domain
        freq_features = freq_domain_analyses.extract_freq_features(
            window, self.fs, bands=bands
        )

        # Stability (Allan deviation)
        adev_features = {}
        if include_adev and n >= adev_min_samples:
            try:
                adev_features = allan_dev.get_allan_deviation(window, self.fs)
            except (ValueError, IndexError, ZeroDivisionError):
                adev_features = {}

        # Wavelet domain (MODWT) — sym4 filter needs >= 8 samples
        wavelet_features = {}
        if include_wavelet and n >= 8:
            try:
                wavelet_features = extract_wavelet_features(
                    window, self.fs, level=wavelet_level
                )
            except (ValueError, IndexError):
                wavelet_features = {}

        # Signal quality
        sq_features = {}
        if include_signal_quality and n >= 8:
            try:
                sq_features = extract_signal_quality_features(window, self.fs)
            except (ValueError, IndexError):
                sq_features = {}

        # Divergence (requires reference distributions)
        div_features = {}
        if include_divergence and divergence_ref_dists is not None:
            try:
                div_features = compute_window_divergence(
                    window, divergence_ref_dists,
                    level=wavelet_level or 5,
                )
            except (ValueError, IndexError):
                div_features = {}

        return {**stats, **freq_features, **adev_features,
                **wavelet_features, **sq_features, **div_features}

    def process_signal(
        self,
        signal: np.ndarray,
        window_size_sec: float,
        step_size_sec: float,
        bands: Optional[Iterable[Sequence[float]]] = None,
        normalize: bool = False,
        normalization_stats: Optional[Mapping[str, float]] = None,
        include_adev: bool = True,
        adev_min_samples: int = 50,
        include_wavelet: bool = True,
        wavelet_level: Optional[int] = 5,
        include_signal_quality: bool = True,
        include_divergence: bool = False,
        divergence_ref_dists: Optional[Dict[str, Tuple]] = None,
    ) -> pd.DataFrame:
        """Sliding window feature extraction. Returns DataFrame (rows=windows)."""
        window_length = int(window_size_sec * self.fs)
        step_length = int(step_size_sec * self.fs)

        if window_length <= 0 or step_length <= 0:
            raise ValueError("Window and step sizes must be positive.")

        values = np.asarray(signal, dtype=np.float64).flatten()
        total_samples = len(values)

        if total_samples < window_length:
            return pd.DataFrame()

        # Prepare normalization parameters
        mean_val = None
        std_val = None
        if normalize:
            mean_val = normalization_stats.get('mean', np.mean(values)) if normalization_stats else np.mean(values)
            std_val = normalization_stats.get('std', np.std(values)) if normalization_stats else np.std(values)
            if std_val == 0:
                std_val = 1.0

        features_list = []

        for i in range(0, total_samples - window_length + 1, step_length):
            window = values[i : i + window_length]

            if normalize:
                window = (window - mean_val) / std_val

            row = self._extract_features(
                window, bands=bands,
                include_adev=include_adev, adev_min_samples=adev_min_samples,
                include_wavelet=include_wavelet, wavelet_level=wavelet_level,
                include_signal_quality=include_signal_quality,
                include_divergence=include_divergence,
                divergence_ref_dists=divergence_ref_dists,
            )

            row['window_start_sample'] = i
            row['window_start_sec'] = i / self.fs

            features_list.append(row)

        df = pd.DataFrame(features_list)

        # Reorder columns to put metadata first
        if len(df) > 0:
            meta_cols = ['window_start_sample', 'window_start_sec']
            other_cols = [c for c in df.columns if c not in meta_cols]
            df = df[meta_cols + other_cols]

        return df

    def process_signal_full(
        self,
        signal: np.ndarray,
        bands: Optional[Iterable[Sequence[float]]] = None,
        include_wavelet: bool = True,
        wavelet_level: Optional[int] = 5,
        include_signal_quality: bool = True,
    ) -> dict:
        """Extract features from the entire signal (no windowing). Returns dict."""
        values = np.asarray(signal, dtype=np.float64).flatten()
        return self._extract_features(
            values, bands=bands,
            include_wavelet=include_wavelet, wavelet_level=wavelet_level,
            include_signal_quality=include_signal_quality,
        )


def extract_features_batch(
    signals: Sequence[np.ndarray],
    fs: float,
    window_size_sec: float,
    step_size_sec: float,
    labels: Optional[Sequence[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """Extract features from multiple signals. Returns DataFrame with 'signal_id' column."""
    extractor = FeatureExtractor(fs=fs)
    all_dfs = []

    for idx, sig in enumerate(signals):
        df = extractor.process_signal(sig, window_size_sec, step_size_sec, **kwargs)
        df['signal_id'] = idx
        if labels is not None and idx < len(labels):
            df['label'] = labels[idx]
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)
