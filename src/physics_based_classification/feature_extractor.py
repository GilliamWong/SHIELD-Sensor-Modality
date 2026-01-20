import numpy as np
import pandas as pd
from typing import Iterable, Mapping, Optional, Sequence

from . import freq_domain_analyses
from . import time_domain_analyses
from . import allan_dev


class FeatureExtractor:
    def __init__(self, fs: float):
        self.fs = fs
    
    #sliding window segmentation of data and feature extraction for each window, returns a DataFrame where rows = windows, columns = feature names 
    #args:
    #   signal: Input signal array (1D or 2D with shape (n_samples, 1)).
    #   window_size_sec: Window size in seconds.
    #   step_size_sec: Step/hop size in seconds.
    #   bands: Optional frequency bands for band energy calculation.
    #   normalize: Whether to normalize each window.
    #   normalization_stats: Optional dict with 'mean' and 'std' for normalization.
    #   include_adev: Whether to include Allan deviation features.
    #   adev_min_samples: Minimum samples required to compute ADEV.
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
    ) -> pd.DataFrame:
        # Convert seconds to samples
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
        window_count = 0
        
        for i in range(0, total_samples - window_length + 1, step_length):
            # Slice the window
            window = values[i : i + window_length]

            if normalize:
                window = (window - mean_val) / std_val
            
            #time domain features
            stats = time_domain_analyses.get_statistical_moments(window)

            #frequency domain features
            freq_features = freq_domain_analyses.extract_freq_features(window, self.fs, bands=bands)

            #stability features (allan deviation)
            adev_features = {}
            if include_adev and window_length >= adev_min_samples:
                try:
                    adev_features = allan_dev.get_allan_deviation(window, self.fs)
                except Exception:
                    # ADEV can fail on very short or constant signals
                    adev_features = {}
            
            # Merge all features into one row
            row = {**stats, **freq_features, **adev_features}
            
            # Add window metadata
            row['window_start_sample'] = i
            row['window_start_sec'] = i / self.fs
            
            features_list.append(row)
            window_count += 1
        
        df = pd.DataFrame(features_list)
        
        # Reorder columns to put metadata first
        if len(df) > 0:
            meta_cols = ['window_start_sample', 'window_start_sec']
            other_cols = [c for c in df.columns if c not in meta_cols]
            df = df[meta_cols + other_cols]
        
        return df
    
    #extract features from the entire signal (no windowing)
    #args:
    #   signal: Input signal array.
    #   bands: Optional frequency bands for band energy calculation.
    #returns:
    #   Dictionary of features.
    def process_signal_full(
        self,
        signal: np.ndarray,
        bands: Optional[Iterable[Sequence[float]]] = None,
    ) -> dict:
        values = np.asarray(signal, dtype=np.float64).flatten()
        
        # Time domain
        stats = time_domain_analyses.get_statistical_moments(values)

        # Frequency domain
        freq_features = freq_domain_analyses.extract_freq_features(values, self.fs, bands=bands)

        # Stability (ADEV)
        adev_features = {}
        if len(values) >= 50:
            try:
                adev_features = allan_dev.get_allan_deviation(values, self.fs)
            except Exception:
                pass
        
        return {**stats, **freq_features, **adev_features}

#convenience function to extract features from multiple signals
    #args:
    #   signals: List of signal arrays.
    #   fs: Sampling frequency in Hz.
def extract_features_batch(
    signals: Sequence[np.ndarray],
    fs: float,
    window_size_sec: float,
    step_size_sec: float,
    labels: Optional[Sequence[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    window_size_sec: Window size in seconds.
    step_size_sec: Step size in seconds.
    labels: Optional list of labels for each signal.
    **kwargs: Additional arguments passed to process_signal.

    Returns:
        DataFrame with features from all signals, with 'signal_id' column.
    """
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