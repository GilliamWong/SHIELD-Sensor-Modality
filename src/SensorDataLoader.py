import numpy as np
import pandas as pd
from scipy.io import loadmat
from typing import Union, Tuple

# loads data, hopefully agnostic to format
# also generates synthetic data for testing, if needed
class SensorDataLoader:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    
    #generates white, pink, or brown noise based on num samples, channels, and noise type
    #note that in this case num_samples represents the number of timesteps (aka, num of samples)
    def generate_synthetic_data(self, num_samples: int, dim: int, mean: float = 0.0, std: float = 1.0, noise: str = "white") -> np.ndarray:
        valid_types = ["white", "pink", "brown"]
        if noise not in valid_types:
            raise ValueError(f"Noise type must be one of {valid_types}")

        white_noise = self.rng.normal(loc=mean, scale=std, size=(num_samples, dim))
        signal = None

        if noise == "white": #random sample from distribution
            signal = white_noise
        elif noise == "brown": #cumulative sum of white noise, random walk where each step is a random sample from the white noise distribution
            signal = np.cumsum(white_noise, axis=0)
        elif noise == "pink": #frequency domain filtering
            X_white = np.fft.rfft(white_noise, axis=0)
            
            frequencies = np.fft.rfftfreq(num_samples)
            
            scaling = np.ones_like(frequencies)
            
            with np.errstate(divide='ignore'):
                scaling[1:] = 1 / np.sqrt(frequencies[1:])
            
            scaling[0] = 0
            
            if dim > 1:
                scaling = scaling[:, np.newaxis]
            
            X_pink = X_white * scaling
            
            signal = np.fft.irfft(X_pink, n=num_samples, axis=0)
        else:
            raise ValueError(f"Unknown noise type: {noise}")

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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"NPZ file not found: {filepath}")
            
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
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MAT file not found: {filepath}")
            
        try:
            mat = loadmat(filepath)
            if variable_name not in mat:
                available = [k for k in mat.keys() if not k.startswith('__')]
                raise KeyError(f"Variable '{variable_name}' not found. Available vars: {available}")
            return mat[variable_name]
        except Exception as e:
            raise IOError(f"Failed to load MATLAB file: {e}")

    