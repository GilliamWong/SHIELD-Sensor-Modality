from testing import prototype_classifier, analyze_feature_discrimination
import sys
import numpy as np
import pandas as pd
from SensorDataLoader import SensorDataLoader
from feature_extractor import FeatureExtractor

#inject fault into real sensor data, adds degradation on top of signal
def inject_fault(signal: np.ndarray, fault_type: str, severity: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(signal)
    t = np.linspace(0, severity, n)
    sig_std = np.std(signal)
    
    if fault_type == 'noise_increase':
        noise = rng.normal(0, 1, n)
        return signal + noise * (3 * t * sig_std)
    
    elif fault_type == 'bias_drift':
        return signal + 2.0 * sig_std * t ** 2
    
    elif fault_type == 'spike_injection':
        out = signal.copy()
        spike_prob = t * 0.02
        spikes = rng.random(n) < spike_prob
        out[spikes] += rng.choice([-1, 1], size=np.sum(spikes)) * 5 * sig_std
        return out
    
    elif fault_type == 'bandwidth_loss':
        from scipy.ndimage import uniform_filter1d
        out = np.zeros_like(signal)
        chunk = 1000
        for i in range(0, n, chunk):
            end = min(i + chunk, n)
            prog = t[min(i + chunk//2, n-1)]
            width = int(1 + 20 * prog)
            out[i:end] = uniform_filter1d(signal[i:end], size=max(1, width))
        return out
    
    elif fault_type == 'saturation':
        out = signal.copy()
        base = 3 * sig_std
        thresh = base * (1 - 0.8 * t)
        return np.clip(out, -thresh, thresh)
    
    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

#build modality classification dataset from PAMAP2
def build_pamap2_modality_dataset(data_dir: str, fs: float = 100.0) -> pd.DataFrame:
    loader = SensorDataLoader()
    extractor = FeatureExtractor(fs=fs)
    
    import glob
    files = glob.glob(f"{data_dir}/*.dat")
    
    all_features = []
    
    modality_map = {
        'accel_16g': 'accelerometer',
        'accel_6g': 'accelerometer',
        'gyro': 'gyroscope',
        'mag': 'magnetometer',
        'temp': 'temperature'
    }
    
    for filepath in files:
        print(f"Processing {filepath}...")
        sensors = loader.load_pamap2(filepath)
        sensors = loader.get_stationary_segments(sensors, activities=[2, 3])
        
        for location in ['hand', 'chest', 'ankle']:
            for sensor_type, modality in modality_map.items():
                key = f'{location}_{sensor_type}'
                data = sensors[key]
                
                # Handle NaNs and check length
                if data.ndim == 1:
                    data = data[~np.isnan(data)]
                    if len(data) < 1000:
                        continue
                    
                    df = extractor.process_signal(data, window_size_sec=2.0, 
                                                   step_size_sec=1.0)
                    df['modality'] = modality
                    df['sensor'] = key
                    df['location'] = location
                    all_features.append(df)
                else:
                    for axis in range(data.shape[1]):
                        sig = data[:, axis]
                        sig = sig[~np.isnan(sig)]
                        if len(sig) < 1000:
                            continue
                        
                        df = extractor.process_signal(sig, window_size_sec=2.0,
                                                       step_size_sec=1.0)
                        df['modality'] = modality
                        df['sensor'] = key
                        df['axis'] = axis
                        df['location'] = location
                        all_features.append(df)
    
    return pd.concat(all_features, ignore_index=True)

#build fault detection dataset: healthy vs degraded signals
def build_fault_detection_dataset(data_dir: str, fs: float = 100.0) -> pd.DataFrame:
    loader = SensorDataLoader()
    extractor = FeatureExtractor(fs=fs)
    
    import glob
    files = glob.glob(f"{data_dir}/*.dat")
    
    fault_types = ['noise_increase', 'bias_drift', 'spike_injection', 
                   'bandwidth_loss', 'saturation']
    
    all_features = []
    
    for filepath in files[:2]:  # Start with 2 files for testing
        print(f"Processing {filepath}...")
        sensors = loader.load_pamap2(filepath)
        sensors = loader.get_stationary_segments(sensors, activities=[2, 3])
        
        # Just use accelerometer for now
        for location in ['hand', 'chest', 'ankle']:
            key = f'{location}_accel_16g'
            data = sensors[key]
            
            for axis in range(data.shape[1]):
                sig = data[:, axis]
                sig = sig[~np.isnan(sig)]
                if len(sig) < 5000:
                    continue
                
                # Healthy
                df = extractor.process_signal(sig, window_size_sec=2.0,
                                               step_size_sec=1.0)
                df['label'] = 'healthy'
                df['fault_type'] = 'none'
                df['sensor'] = key
                all_features.append(df)
                
                # Faulty versions
                for fault in fault_types:
                    faulty = inject_fault(sig, fault, severity=1.0)
                    df = extractor.process_signal(faulty, window_size_sec=2.0,
                                                   step_size_sec=1.0)
                    df['label'] = 'degraded'
                    df['fault_type'] = fault
                    df['sensor'] = key
                    all_features.append(df)
    
    return pd.concat(all_features, ignore_index=True)

#run modality experiment, classifies sensor modality from real PAMAP2 data
def run_modality_experiment(data_dir: str):
    print("=" * 70)
    print("MODALITY DETECTION - REAL PAMAP2 DATA")
    print("=" * 70)
    
    # Build dataset
    print("\n[1/3] Building feature dataset from PAMAP2...")
    df = build_pamap2_modality_dataset(data_dir, fs=100.0)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Modalities: {df['modality'].unique()}")
    print(f"Samples per modality:\n{df['modality'].value_counts()}")
    
    # Analyze which features discriminate best
    print("\n[2/3] Analyzing feature discrimination...")
    discrimination = analyze_feature_discrimination(df,
                                                     label_col='modality')
    
    print("\nTop 10 discriminative features:")
    print(discrimination.head(10))
    
    # Train classifier
    print("\n[3/3] Training Random Forest classifier...")
    clf, le, importances = prototype_classifier(df, label_col='modality')
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy: {clf.test_accuracy_:.2%}")
    print(f"Most important feature: {importances.iloc[0]['feature']}")
    print(f"Most discriminative feature: {discrimination.iloc[0]['feature']}")
    
    return df, clf, discrimination

#run fault detection experiment, classifies healthy vs degraded signals
def run_fault_detection_experiment(data_dir: str):
    print("=" * 70)
    print("FAULT DETECTION - SYNTHETIC DEGRADATION")
    print("=" * 70)
    
    # Build dataset with healthy + degraded signals
    print("\n[1/2] Building fault detection dataset...")
    df = build_fault_detection_dataset(data_dir, fs=100.0)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Labels: {df['label'].unique()}")
    print(f"Samples per label:\n{df['label'].value_counts()}")
    print(f"Fault types:\n{df['fault_type'].value_counts()}")
    
    # Train binary classifier: healthy vs degraded
    print("\n[2/2] Training fault detector...")
    clf, le, importances = prototype_classifier(df, label_col='label')
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy: {clf.test_accuracy_:.2%}")
    print(f"Most important feature: {importances.iloc[0]['feature']}")
    
    return df, clf

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pamap2_testing.py <path_to_pamap2_data>")
        print("Example: python pamap2_testing.py ./PAMAP2_Dataset/Protocol")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    run_modality_experiment(data_dir)
    run_fault_detection_experiment(data_dir)