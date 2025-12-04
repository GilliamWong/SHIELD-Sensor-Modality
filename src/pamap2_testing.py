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