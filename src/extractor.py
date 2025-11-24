import numpy as np
import pandas as pd
from typing import Iterable, Mapping, Optional, Sequence

import freq_domain
import time_domain

class FeatureExtractor:
    def __init__(self, fs: float):
        self.fs = fs
    
    #applies sliding window segmentation and extracts features for each window.
    #returns a DataFrame where rows = windows, columns = feature names.
    def process_signal(
        self,
        signal: np.ndarray,
        window_size_sec: float,
        step_size_sec: float,
        bands: Optional[Iterable[Sequence[float]]] = None,
        normalize: bool = False,
        normalization_stats: Optional[Mapping[str, float]] = None,
    ) -> pd.DataFrame:
        #convert seconds to samples
        window_length = int(window_size_sec * self.fs)
        step_length = int(step_size_sec * self.fs)

        if window_length <= 0 or step_length <= 0:
            raise ValueError("Window and step sizes must be positive.")
        
        values = np.asarray(signal, dtype=np.float64).flatten()
        total_samples = len(values)

        if total_samples < window_length:
            return pd.DataFrame()

        mean_val = None
        std_val = None
        if normalize:
            if normalization_stats:
                mean_val = float(normalization_stats.get('mean', np.mean(values)))
                std_val = float(normalization_stats.get('std', np.std(values)))
            else:
                mean_val = float(np.mean(values))
                std_val = float(np.std(values))
            if std_val == 0:
                std_val = 1.0
        
        features_list = []
        for i in range(0, total_samples - window_length + 1, step_length):
            #slice the window
            window = values[i : i + window_length]

            if normalize:
                window = (window - mean_val) / std_val
            
            #extract features
            stats = time_domain.get_statistical_moments(window)
            freq_features = freq_domain.extract_freq_features(window, self.fs, bands=bands)
            
            #merge dictionaries into one row
            row = {**stats, **freq_features}
            
            features_list.append(row)
            
        return pd.DataFrame(features_list)  
