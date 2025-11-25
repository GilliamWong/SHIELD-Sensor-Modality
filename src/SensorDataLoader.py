import os
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Dict, Optional, Sequence, Tuple, Union

# loads data, hopefully agnostic to format
# also generates synthetic data for testing, if needed
class SensorDataLoader:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def _infer_sample_rate(self, timestamps: Sequence[float]) -> Optional[float]:
        if len(timestamps) < 2:
            return None
        deltas = np.diff(timestamps)
        deltas = deltas[deltas > 0]
        if len(deltas) == 0:
            return None
        median_delta = float(np.median(deltas))
        if median_delta <= 0:
            return None
        return 1.0 / median_delta

    #generates white, pink, or brown noise based on num samples, channels, and noise type
    #note that in this case num_samples represents the number of timesteps (aka, num of samples)
    def generate_synthetic_data(
        self,
        num_samples: int,
        dim: int = 1,
        mean: float = 0.0,
        std: float = 1.0,
        noise_type: str = "white",
    ) -> np.ndarray:
        valid_types = ["white", "pink", "brown"]
        if noise_type not in valid_types:
            raise ValueError(f"Noise type must be one of {valid_types}")

        white_noise = self.rng.normal(loc=mean, scale=std, size=(num_samples, dim))
        signal = None

        if noise_type == "white": #random sample from distribution
            signal = white_noise
        elif noise_type == "brown": #cumulative sum of white noise, random walk where each step is a random sample from the white noise distribution
            signal = np.cumsum(white_noise, axis=0)
        elif noise_type == "pink": #frequency domain filtering
            X_white = np.fft.rfft(white_noise, axis=0)
            
            frequencies = np.fft.rfftfreq(num_samples)
            
            scaling = np.ones_like(frequencies)
            
            with np.errstate(divide='ignore'):
                scaling[1:] = 1 / np.sqrt(frequencies[1:])
            
            scaling[0] = 0
            scaling = scaling[:, np.newaxis]
            
            X_pink = X_white * scaling
            
            signal = np.fft.irfft(X_pink, n=num_samples, axis=0)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        #rescale to match mean and std, integration and filtering change the signal's mean and std
        signal = signal - np.mean(signal, axis=0) 
        current_std = np.std(signal, axis=0)
        
        #no div by zero
        current_std[current_std == 0] = 1.0 
        
        signal = (signal / current_std) * std + mean
        
        return signal
    
    #loads nasa cmapss data, space delimited, 26 columns (Unit, Time, Settings 1-3, Sensors 1-21)
    #sensor_indices (list): List of 0-based column indices to extract. If None, returns all 21 sensors (cols 5-26).
    def load_cmapss_txt(self, filepath: str, sensor_indices: list = None) -> np.ndarray:
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
            
            if sensor_indices is None:
                data = df.iloc[:, 5:26].values
            else:
                data = df.iloc[:, sensor_indices].values
                
            return data.astype(np.float32)
            
        except Exception as e:
            raise IOError(f"Failed to parse C-MAPSS file {filepath}: {e}")
    
    #loads npz file, compressed, typed arrays and metadata
    def load_npz(self, filepath: str, key: str = 'signal') -> np.ndarray:
        try:
            with np.load(filepath) as data:
                if key not in data:
                    available = list(data.keys())
                    raise KeyError(f"Key '{key}' not found in NPZ. Available keys: {available}")
                return data[key]
        except Exception as e:
            raise IOError(f"Failed to load NPZ: {e}")

    #loads mat file, legacy engineering data from MATLAB .mat files (e.g., UNSW Bearing Data)
    def load_matlab(self, filepath: str, variable_name: str) -> np.ndarray:
        try:
            mat = loadmat(filepath)
            if variable_name not in mat:
                available = [k for k in mat.keys() if not k.startswith('__')]
                raise KeyError(f"Variable '{variable_name}' not found. Available vars: {available}")
            return mat[variable_name]
        except Exception as e:
            raise IOError(f"Failed to load MATLAB file: {e}")

    def load_tabular(
        self,
        filepath: str,
        time_column: Optional[str] = None,
        channel_columns: Optional[Sequence[str]] = None,
        sample_rate: Optional[float] = None,
    ) -> Tuple[np.ndarray, Optional[float]]:
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".csv", ".txt"]:
            df = pd.read_csv(filepath)
        elif ext in [".parquet", ".pq"]:
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported tabular format for {filepath}")

        if channel_columns is None:
            channel_columns = [c for c in df.columns if c != time_column]

        data = df[channel_columns].to_numpy(dtype=np.float32)

        inferred_rate = sample_rate
        if time_column is not None and sample_rate is None:
            inferred_rate = self._infer_sample_rate(df[time_column].to_numpy())
            if inferred_rate is None:
                warnings.warn("Unable to infer sample rate from time column; provide sample_rate explicitly.", RuntimeWarning)

        return data, inferred_rate

    
