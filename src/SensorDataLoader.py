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

    #loads pamap2 data, 5 columns (timestamp, activity, heart_rate, hand_temp, chest_temp, ankle_temp)
    def load_pamap2(self, filepath: str) -> dict:
        data = pd.read_csv(filepath, sep=r'\s+', header=None, 
                        na_values='NaN', engine='python')
        
        sensors = {
            'timestamp': data.iloc[:, 0].values,
            'activity': data.iloc[:, 1].values.astype(int),
            'heart_rate': data.iloc[:, 2].values,
        }
        
        # IMU offsets: hand=3, chest=20, ankle=37
        imu_config = {'hand': 3, 'chest': 20, 'ankle': 37}
        
        for location, offset in imu_config.items():
            sensors[f'{location}_temp'] = data.iloc[:, offset].values
            sensors[f'{location}_accel_16g'] = data.iloc[:, offset+1:offset+4].values
            sensors[f'{location}_accel_6g'] = data.iloc[:, offset+4:offset+7].values
            sensors[f'{location}_gyro'] = data.iloc[:, offset+7:offset+10].values
            sensors[f'{location}_mag'] = data.iloc[:, offset+10:offset+13].values
        
        return sensors

    #filters out non-stationary segments (e.g., walking, running)
    def get_stationary_segments(self, sensors: dict, activities: list = [2, 3]) -> dict:
        mask = np.isin(sensors['activity'], activities)
        
        filtered = {}
        for key, val in sensors.items():
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    filtered[key] = val[mask]
                else:
                    filtered[key] = val[mask, :]
        
        return filtered

    def load_uci_har(self, dataset_dir: str, split: str = 'train') -> dict:
        """Load UCI HAR dataset (smartphone accelerometer + gyroscope at 50 Hz).

        Args:
            dataset_dir: Path to 'UCI HAR Dataset' root containing train/ and test/ folders.
            split: 'train' or 'test'.

        Returns:
            dict with keys: total_acc_x, total_acc_y, total_acc_z,
            body_gyro_x, body_gyro_y, body_gyro_z, subject, activity.
            Signal arrays are 1D (all 128-sample windows concatenated).
        """
        signal_dir = os.path.join(dataset_dir, split, 'Inertial Signals')

        signal_files = {
            'total_acc_x': f'total_acc_x_{split}.txt',
            'total_acc_y': f'total_acc_y_{split}.txt',
            'total_acc_z': f'total_acc_z_{split}.txt',
            'body_gyro_x': f'body_gyro_x_{split}.txt',
            'body_gyro_y': f'body_gyro_y_{split}.txt',
            'body_gyro_z': f'body_gyro_z_{split}.txt',
        }

        result = {}
        for key, filename in signal_files.items():
            filepath = os.path.join(signal_dir, filename)
            data = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
            result[key] = data.values.flatten().astype(np.float64)

        subjects = pd.read_csv(
            os.path.join(dataset_dir, split, f'subject_{split}.txt'),
            header=None,
        ).values.flatten()
        activities = pd.read_csv(
            os.path.join(dataset_dir, split, f'y_{split}.txt'),
            header=None,
        ).values.flatten()

        result['subject'] = np.repeat(subjects, 128)
        result['activity'] = np.repeat(activities, 128)

        return result

    def load_shl_motion(self, filepath: str, max_rows: int = None) -> dict:
        """Load a single SHL Motion file (space-separated, 23 columns at 100 Hz).

        Column layout: timestamp, accel(x,y,z), gyro(x,y,z), mag(x,y,z), ...

        Args:
            filepath: Path to a *_Motion.txt file.
            max_rows: Max rows to load (None = all).

        Returns:
            dict with keys: timestamp, accel_x/y/z, gyro_x/y/z, mag_x/y/z.
        """
        data = np.loadtxt(filepath, max_rows=max_rows)
        nan_mask = ~np.isnan(data).any(axis=1)
        data = data[nan_mask]

        return {
            'timestamp': data[:, 0],
            'accel_x': data[:, 1], 'accel_y': data[:, 2], 'accel_z': data[:, 3],
            'gyro_x': data[:, 4], 'gyro_y': data[:, 5], 'gyro_z': data[:, 6],
            'mag_x': data[:, 7], 'mag_y': data[:, 8], 'mag_z': data[:, 9],
        }

    def load_opportunity(self, filepath: str, column_names_file: str = None) -> dict:
        """Load a single OPPORTUNITY .dat file (space-separated, 30 Hz).

        Extracts IMU accel/gyro/mag from the InertialMeasurementUnit columns.
        OPPORTUNITY IMU columns (0-indexed in data):
            Columns 37-45: BACK IMU (accel 3, gyro 3, mag 3)
            Columns 50-58: RUA (right upper arm) IMU
            Columns 63-71: RLA (right lower arm) IMU
            Columns 76-84: LUA (left upper arm) IMU
            Columns 89-97: LLA (left lower arm) IMU

        Args:
            filepath: Path to an OPPORTUNITY .dat file (e.g., S1-ADL1.dat).
            column_names_file: Optional path to column_names.txt for reference.

        Returns:
            dict mapping '{location}_{modality}_{axis}' -> 1D np.ndarray.
            Also includes 'activity_ml' (column 243, mid-level activity label).
        """
        data = np.loadtxt(filepath)

        imu_locations = {
            'back': 37, 'rua': 50, 'rla': 63, 'lua': 76, 'lla': 89,
        }

        result = {}
        result['timestamp'] = data[:, 0]

        for loc, offset in imu_locations.items():
            result[f'{loc}_accel_x'] = data[:, offset]
            result[f'{loc}_accel_y'] = data[:, offset + 1]
            result[f'{loc}_accel_z'] = data[:, offset + 2]
            result[f'{loc}_gyro_x'] = data[:, offset + 3]
            result[f'{loc}_gyro_y'] = data[:, offset + 4]
            result[f'{loc}_gyro_z'] = data[:, offset + 5]
            result[f'{loc}_mag_x'] = data[:, offset + 6]
            result[f'{loc}_mag_y'] = data[:, offset + 7]
            result[f'{loc}_mag_z'] = data[:, offset + 8]

        # Mid-level activity label (column 243, 0-indexed)
        if data.shape[1] > 243:
            result['activity_ml'] = data[:, 243].astype(int)

        return result

    def load_realworld_zip(self, zip_path: str) -> dict:
        """Load all CSVs from a RealWorld HAR zip file.

        Each zip contains CSVs for 7 body positions with columns:
        id, attr_time, attr_x, attr_y, attr_z.

        Args:
            zip_path: Path to a zip like acc_walking_csv.zip.

        Returns:
            dict mapping body_part -> {x, y, z} arrays.
            E.g. {'chest': {'x': arr, 'y': arr, 'z': arr}, ...}
        """
        import zipfile as _zf
        result = {}
        with _zf.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if not name.endswith('.csv'):
                    continue
                # Parse body part from filename like "acc_walking_chest.csv"
                base = os.path.splitext(os.path.basename(name))[0]
                parts = base.split('_')
                body_part = parts[-1] if len(parts) >= 3 else base
                with zf.open(name) as f:
                    df = pd.read_csv(f)
                    result[body_part] = {
                        'x': df['attr_x'].values.astype(np.float64),
                        'y': df['attr_y'].values.astype(np.float64),
                        'z': df['attr_z'].values.astype(np.float64),
                    }
        return result

