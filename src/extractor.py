import numpy as np
import pandas as pd
from src.features import time_domain, freq_domain

class FeatureExtractor:
    def __init__(self, fs: float):
        self.fs = fs
    
    #applies sliding window segmentation and extracts features for each window.
    #returns a DataFrame where rows = windows, columns = feature names.
    def process_signal(self, signal: np.ndarray, window_size_sec: float, step_size_sec: float) -> pd.DataFrame:
        #convert seconds to samples
        window_length = int(window_size_sec * self.fs)
        step_length = int(step_size_sec * self.fs)
        
        features_list = []
        for i in range(0, len(signal) - window_length + 1, step_length):
            #slice the window
            window = signal[i : i + window_length]
            
            #extract features
            stats = time_domain.get_statistical_moments(window)
            freq_features = freq_domain.extract_freq_features(window, self.fs)
            
            #merge dictionaries into one row
            row = {**stats, **freq_features}
            
            features_list.append(row)
            
        return pd.DataFrame(features_list)  