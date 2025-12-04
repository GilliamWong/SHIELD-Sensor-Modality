"""
Project SHIELD - Comprehensive Testing and Analysis Suite (v2)

This module provides:
1. Physics validation - Verify pipeline correctly identifies noise types
2. Sensor modality simulation - Create synthetic sensor signatures
3. Degradation simulation - Model sensor failure modes
4. Feature discrimination analysis - Identify most useful features
5. Prototype classifier - End-to-end pipeline validation

Author: William 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter1d
from typing import Dict, List, Tuple, Optional
import warnings

from SensorDataLoader import SensorDataLoader
from feature_extractor import FeatureExtractor
from freq_domain_analyses import get_psd_welch, get_spectral_slope

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Metadata columns to exclude from feature analysis
METADATA_COLUMNS = [
    'instance_id', 'noise_type', 'window_start_sample',
    'window_start_sec', 'degradation_progress', 'degradation_type',
    'signal_id', 'label', 'sensor', 'location', 'axis', 'fault_type'
]


# =============================================================================
# 1. PHYSICS VALIDATION
# =============================================================================

def validate_physics(save_plot: bool = True) -> Dict[str, dict]:
    """
    End-to-End Verification of the SHIELD Feature Extraction Pipeline.
    
    1. Generates synthetic sensor data (White, Pink, Brown).
    2. Runs the full extraction pipeline (Sliding Window).
    3. Asserts that extracted features match physical laws.
    4. Generates a visual report (PNG).
    
    Returns:
        Dictionary with validation results for each noise type.
    """
    # Configuration
    FS = 100.0          # 100 Hz sampling
    DURATION = 100      # 100 seconds of data
    N_SAMPLES = int(FS * DURATION)
    WINDOW_SEC = 1.0    # 1 second windows
    STEP_SEC = 0.5      # 50% overlap
    
    loader = SensorDataLoader(seed=42)
    extractor = FeatureExtractor(fs=FS)
    
    print("=" * 60)
    print("SHIELD PHYSICS VALIDATION")
    print(f"Parameters: fs={FS}Hz, N={N_SAMPLES}, window={WINDOW_SEC}s")
    print("=" * 60)

    # Generate Synthetic Signals
    print("\nGenerating synthetic noise profiles...")
    signals = {
        'White (Accel)': loader.generate_synthetic_data(N_SAMPLES, noise_type='white'),
        'Pink (Elec)':   loader.generate_synthetic_data(N_SAMPLES, noise_type='pink'),
        'Brown (Gyro)':  loader.generate_synthetic_data(N_SAMPLES, noise_type='brown')
    }

    # Process & Validate
    results = {}
    all_passed = True
    
    for name, signal in signals.items():
        print(f"\nProcessing {name}...")
        
        # Run the pipeline
        df = extractor.process_signal(signal.flatten(), WINDOW_SEC, STEP_SEC)
        
        # Calculate aggregate metrics
        window_slope = df['psd_slope'].mean()
        avg_entropy = df['spectral_entropy'].mean()
        avg_flatness = df['spectral_flatness'].mean()

        # Full-signal PSD for ground-truth slope
        f_full, p_full = get_psd_welch(signal.flatten(), fs=FS)
        full_slope = get_spectral_slope(f_full, p_full)
        
        results[name] = {
            'window_slope': window_slope,
            'full_slope': full_slope,
            'entropy': avg_entropy,
            'flatness': avg_flatness,
            'n_windows': len(df),
            'n_features': len(df.columns),
        }
        
        print(f"  -> Extracted {len(df)} windows, {len(df.columns)} features each")
        print(f"  -> Avg PSD Slope (windowed): {window_slope:.4f}")
        print(f"  -> Full-signal PSD Slope:    {full_slope:.4f}")
        print(f"  -> Avg Spectral Entropy:     {avg_entropy:.4f}")
        print(f"  -> Avg Spectral Flatness:    {avg_flatness:.4f}")
        
        # Physics Assertions
        slope_for_check = full_slope if np.isfinite(full_slope) else window_slope
        passed = False
        
        if 'White' in name:
            passed = -0.5 < slope_for_check < 0.5
            expected = "~0"
        elif 'Pink' in name:
            passed = -1.5 < slope_for_check < -0.5
            expected = "~-1"
        elif 'Brown' in name:
            passed = slope_for_check < -1.5
            expected = "< -1.5"
        
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} Slope is {expected} (actual: {slope_for_check:.2f})")
        results[name]['passed'] = passed
        
        if not passed:
            all_passed = False

    # Generate Visualization
    if save_plot:
        print("\nGenerating visual report...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: PSDs
        ax1 = axes[0]
        for name, signal in signals.items():
            f, p = get_psd_welch(signal.flatten(), fs=FS)
            mask = f > 0
            slope_label = results[name]['full_slope']
            if not np.isfinite(slope_label):
                slope_label = results[name]['window_slope']
            ax1.loglog(f[mask], p[mask], label=f"{name} (slope={slope_label:.2f})", alpha=0.8)

        ax1.set_title("Power Spectral Density - Noise Color Verification")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Power Spectral Density (V²/Hz)")
        ax1.legend()
        ax1.grid(True, which="both", ls="-", alpha=0.5)
        
        # Plot 2: Feature Comparison
        ax2 = axes[1]
        noise_types = list(results.keys())
        x = np.arange(len(noise_types))
        width = 0.25
        
        slopes = [results[n]['window_slope'] for n in noise_types]
        entropies = [results[n]['entropy'] for n in noise_types]
        flatnesses = [results[n]['flatness'] for n in noise_types]
        
        ax2.bar(x - width, slopes, width, label='PSD Slope', color='steelblue')
        ax2.bar(x, entropies, width, label='Spectral Entropy', color='coral')
        ax2.bar(x + width, flatnesses, width, label='Spectral Flatness', color='seagreen')
        
        ax2.set_ylabel('Feature Value')
        ax2.set_title('Feature Comparison Across Noise Types')
        ax2.set_xticks(x)
        ax2.set_xticklabels([n.split()[0] for n in noise_types])
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('physics_validation.png', dpi=150)
        print("Saved: physics_validation.png")
        plt.close()
    
    print("\n" + "=" * 60)
    print(f"VALIDATION {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 60)
    
    return results


# =============================================================================
# 2. SENSOR MODALITY SIMULATION
# =============================================================================

def simulate_sensor_modalities(
    n_samples: int = 10000,
    fs: float = 100.0,
    n_instances_per_class: int = 5,
    seed: int = 42,
    zero_mean: bool = True  # NEW: Remove bias for cleaner noise analysis
) -> pd.DataFrame:
    """
    Simulate different sensor modalities with characteristic noise profiles.
    Creates multiple instances of each sensor type with slight variations.
    
    Args:
        n_samples: Number of samples per signal.
        fs: Sampling frequency in Hz.
        n_instances_per_class: Number of signal instances per sensor type.
        seed: Random seed for reproducibility.
        zero_mean: If True, don't add bias (better for noise classification).
        
    Returns:
        DataFrame with extracted features and sensor labels.
    """
    loader = SensorDataLoader(seed=seed)
    extractor = FeatureExtractor(fs=fs)
    rng = np.random.default_rng(seed)
    
    # Define sensor profiles based on typical noise characteristics
    # Key insight: Different sensors have different NOISE COLORS
    sensor_profiles = {
        'accelerometer': {
            'noise_type': 'white',
            'base_std': 1.0,  # Normalized for comparison
            'description': 'White noise dominated (velocity random walk)',
        },
        'gyroscope': {
            'noise_type': 'pink',
            'base_std': 1.0,
            'description': 'Pink/flicker noise (bias instability region)',
        },
        'pressure': {
            'noise_type': 'brown',
            'base_std': 1.0,
            'description': 'Brown noise (slow atmospheric variations)',
        },
        'temperature': {
            'noise_type': 'brown',
            'base_std': 0.8,  # Slightly different std
            'description': 'Brown noise (thermal drift)',
        },
        'magnetometer': {
            'noise_type': 'pink',
            'base_std': 1.2,  # Slightly different std
            'description': 'Pink noise with environmental interference',
        },
    }
    
    print("=" * 60)
    print("SENSOR MODALITY SIMULATION")
    print(f"Parameters: {n_samples} samples, {fs}Hz, {n_instances_per_class} instances/class")
    print(f"Zero-mean mode: {zero_mean}")
    print("=" * 60)
    
    all_features = []
    
    for sensor_name, profile in sensor_profiles.items():
        print(f"\nSimulating {sensor_name}... ({profile['description']})")
        
        for instance in range(n_instances_per_class):
            # Add variation between instances (±20% std variation)
            std_variation = profile['base_std'] * (1 + 0.2 * rng.standard_normal())
            std_variation = max(0.1, std_variation)  # Ensure positive
            
            # Generate base noise
            signal = loader.generate_synthetic_data(
                n_samples, 
                noise_type=profile['noise_type'],
                std=std_variation
            ).flatten()
            
            # Add slight drift for gyroscope/magnetometer (characteristic behavior)
            if sensor_name in ['gyroscope', 'magnetometer']:
                drift_rate = rng.uniform(-1e-4, 1e-4)
                signal += drift_rate * np.arange(n_samples)
            
            # Optionally add realistic bias (but this makes RMS/mean dominate)
            if not zero_mean:
                bias_ranges = {
                    'accelerometer': (-0.1, 0.1),
                    'gyroscope': (-0.05, 0.05),
                    'pressure': (1010, 1020),
                    'temperature': (20, 30),
                    'magnetometer': (-50, 50),
                }
                bias = rng.uniform(*bias_ranges[sensor_name])
                signal += bias
            
            # Extract features
            df = extractor.process_signal(
                signal, 
                window_size_sec=1.0, 
                step_size_sec=0.5,
                include_adev=True
            )
            
            # Add labels
            df['sensor_type'] = sensor_name
            df['instance_id'] = instance
            df['noise_type'] = profile['noise_type']
            
            all_features.append(df)
        
        print(f"  -> Generated {n_instances_per_class} instances")
    
    result_df = pd.concat(all_features, ignore_index=True)
    
    print(f"\nTotal samples: {len(result_df)}")
    print(f"Features per sample: {len([c for c in result_df.columns if c not in ['sensor_type', 'instance_id', 'noise_type', 'window_start_sample', 'window_start_sec']])}")
    
    return result_df


# =============================================================================
# 3. DEGRADATION SIMULATION
# =============================================================================

def simulate_degradation(
    n_samples: int = 50000,
    fs: float = 100.0,
    degradation_type: str = 'noise_increase',
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Simulate a sensor degrading over time.
    Watch how features change as degradation progresses.
    
    Args:
        n_samples: Total number of samples.
        fs: Sampling frequency in Hz.
        degradation_type: One of 'noise_increase', 'bias_drift', 
                         'spike_injection', 'bandwidth_loss'.
        seed: Random seed.
        
    Returns:
        Tuple of (features DataFrame, raw signal array).
    """
    loader = SensorDataLoader(seed=seed)
    extractor = FeatureExtractor(fs=fs)
    rng = np.random.default_rng(seed)
    
    # Start with healthy white noise
    base_signal = loader.generate_synthetic_data(
        n_samples, noise_type='white', std=1.0
    ).flatten()
    
    # Create degradation envelope (0 = healthy, 1 = failed)
    t = np.arange(n_samples) / n_samples
    
    if degradation_type == 'noise_increase':
        # Noise floor gradually increases (common failure mode)
        signal = base_signal * (1 + 3 * t)  # Noise quadruples by end

    elif degradation_type == 'bias_drift':
        # Sensor develops a growing bias (calibration loss)
        signal = base_signal + 2.0 * t ** 2  # Quadratic drift
        
    elif degradation_type == 'spike_injection':
        # Sensor starts producing occasional spikes (intermittent fault)
        signal = base_signal.copy()
        spike_prob = t * 0.02  # Increasing probability of spikes
        spikes = rng.random(n_samples) < spike_prob
        signal[spikes] += rng.choice([-1, 1], size=np.sum(spikes)) * 5
        
    elif degradation_type == 'bandwidth_loss':
        # High-frequency response degrades (mechanical wear)
        # Apply increasing smoothing filter
        signal = np.zeros_like(base_signal)
        chunk_size = 1000
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            progress = (i + chunk_size/2) / n_samples
            filter_width = int(1 + 20 * progress)
            signal[i:end_idx] = uniform_filter1d(
                base_signal[i:end_idx], 
                size=max(1, filter_width)
            )
    
    elif degradation_type == 'saturation':
        # Sensor starts clipping (ADC or physical limit reached)
        signal = base_signal.copy()
        clip_threshold = 3.0 - 2.5 * t  # Threshold decreases over time
        for i in range(n_samples):
            if abs(signal[i]) > clip_threshold[i]:
                signal[i] = np.sign(signal[i]) * clip_threshold[i]
    
    else:
        raise ValueError(f"Unknown degradation type: {degradation_type}")
    
    # Extract features in windows across the degradation trajectory
    df = extractor.process_signal(
        signal, 
        window_size_sec=2.0, 
        step_size_sec=1.0,
        include_adev=True
    )
    
    # Add degradation progress indicator
    df['degradation_progress'] = np.linspace(0, 1, len(df))
    df['degradation_type'] = degradation_type
    
    return df, signal


def analyze_all_degradation_modes(save_plot: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Analyze all degradation modes and visualize feature evolution.
    
    IMPROVEMENT: 
    - Uses smoothing for cleaner trends
    - Excludes noisy kurtosis from main plot
    - Uses separate y-axes or better feature selection
    
    Returns:
        Dictionary mapping degradation type to feature DataFrame.
    """
    degradation_types = [
        'noise_increase', 
        'bias_drift', 
        'spike_injection', 
        'bandwidth_loss',
        'saturation'
    ]
    
    print("=" * 60)
    print("DEGRADATION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    if save_plot:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
    
    for idx, deg_type in enumerate(degradation_types):
        print(f"\nSimulating {deg_type}...")
        df, signal = simulate_degradation(degradation_type=deg_type)
        results[deg_type] = df
        
        # Report key feature changes
        early = df[df['degradation_progress'] < 0.1]
        late = df[df['degradation_progress'] > 0.9]
        
        print(f"  Feature changes (early -> late):")
        print(f"    Variance:  {early['variance'].mean():.4f} -> {late['variance'].mean():.4f}")
        print(f"    RMS:       {early['rms'].mean():.4f} -> {late['rms'].mean():.4f}")
        print(f"    Kurtosis:  {early['kurtosis'].mean():.4f} -> {late['kurtosis'].mean():.4f}")
        print(f"    PSD Slope: {early['psd_slope'].mean():.4f} -> {late['psd_slope'].mean():.4f}")
        
        if save_plot:
            ax = axes[idx]
            
            # IMPROVED: Select stable features and apply smoothing
            # Don't include kurtosis - it's too noisy
            feature_config = [
                ('variance', 'blue', 'Variance'),
                ('rms', 'orange', 'RMS'),
                ('spectral_entropy', 'red', 'Sp. Entropy'),
                ('psd_slope', 'purple', 'PSD Slope'),
            ]
            
            for feat, color, label in feature_config:
                if feat in df.columns:
                    vals = df[feat].values
                    
                    # Apply smoothing (moving average)
                    window_size = max(5, len(vals) // 20)
                    smoothed = uniform_filter1d(vals, size=window_size)
                    
                    # Z-score normalize for comparison (center at 0, scale by std)
                    mean_val = smoothed[0] if smoothed[0] != 0 else 1
                    normalized = (smoothed - smoothed[0]) / (np.std(smoothed) + 1e-10)
                    
                    ax.plot(df['degradation_progress'], normalized, 
                           label=label, alpha=0.8, linewidth=2, color=color)
            
            ax.set_xlabel('Degradation Progress')
            ax.set_ylabel('Normalized Change (σ)')
            ax.set_title(f'{deg_type.replace("_", " ").title()}')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    # Remove empty subplot
    if save_plot:
        axes[-1].axis('off')
        plt.suptitle('Feature Evolution During Sensor Degradation (Smoothed)', fontsize=14)
        plt.tight_layout()
        plt.savefig('degradation_analysis.png', dpi=150)
        print("\nSaved: degradation_analysis.png")
        plt.close()
    
    return results


# =============================================================================
# 4. FEATURE DISCRIMINATION ANALYSIS (IMPROVED)
# =============================================================================

def analyze_feature_discrimination(
    df: pd.DataFrame, 
    label_col: str = 'sensor_type',
    top_n: int = 15
) -> pd.DataFrame:
    """
    Analyze which features best discriminate between sensor types.
    Uses ANOVA F-statistic to rank feature importance.
    
    Args:
        df: DataFrame with features and labels.
        label_col: Name of the label column.
        top_n: Number of top features to display.
        
    Returns:
        DataFrame with feature rankings.
    """
    # Identify feature columns (exclude metadata and labels)
    exclude_cols = METADATA_COLUMNS + [label_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print("=" * 60)
    print("FEATURE DISCRIMINATION ANALYSIS")
    print(f"Analyzing {len(feature_cols)} features across {df[label_col].nunique()} classes")
    print("=" * 60)
    
    results = []
    groups = df.groupby(label_col)
    
    for feat in feature_cols:
        # Get feature values for each class
        class_values = [group[feat].dropna().values for name, group in groups]
        
        # Skip if any class has insufficient samples
        if any(len(v) < 3 for v in class_values):
            continue
        
        # Skip if feature has no variance
        if df[feat].std() < 1e-10:
            continue
            
        try:
            # ANOVA F-test
            f_stat, p_value = stats.f_oneway(*class_values)
            
            if not np.isfinite(f_stat):
                continue
            
            # Effect size (eta-squared)
            grand_mean = df[feat].mean()
            ss_between = sum(len(v) * (np.mean(v) - grand_mean)**2 for v in class_values)
            ss_total = sum((df[feat] - grand_mean)**2)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            # Coefficient of variation between classes
            class_means = [np.mean(v) for v in class_values]
            cv_between = np.std(class_means) / (np.abs(np.mean(class_means)) + 1e-10)
            
            results.append({
                'feature': feat,
                'f_statistic': f_stat,
                'p_value': p_value,
                'eta_squared': eta_squared,
                'cv_between_classes': cv_between,
                'significant': p_value < 0.05,
            })
        except Exception as e:
            continue
    
    results_df = pd.DataFrame(results).sort_values('f_statistic', ascending=False)
    
    print(f"\nTop {top_n} Most Discriminative Features:")
    print("-" * 60)
    print(f"{'Feature':<25} {'F-stat':>10} {'p-value':>12} {'η²':>8}")
    print("-" * 60)
    
    for _, row in results_df.head(top_n).iterrows():
        sig = '*' if row['significant'] else ''
        print(f"{row['feature']:<25} {row['f_statistic']:>10.2f} {row['p_value']:>12.2e} {row['eta_squared']:>8.3f}{sig}")
    
    print("-" * 60)
    print(f"* = statistically significant (p < 0.05)")
    print(f"\nTotal significant features: {results_df['significant'].sum()} / {len(results_df)}")
    
    return results_df


def visualize_feature_distributions(
    df: pd.DataFrame,
    features: List[str] = None,
    label_col: str = 'sensor_type',
    save_plot: bool = True
) -> None:
    """
    Visualize the distribution of top features across sensor types.
    
    IMPROVEMENT: Default to noise-characteristic features instead of
    bias-dominated features like RMS/mean.
    
    Args:
        df: DataFrame with features and labels.
        features: List of feature names to visualize. If None, uses good defaults.
        label_col: Name of the label column.
        save_plot: Whether to save the plot.
    """
    # IMPROVED: Use noise-characteristic features by default
    if features is None:
        # These features characterize noise type, not DC offset
        default_features = [
            'psd_slope',           # Key discriminator: white=0, pink=-1, brown=-2
            'spectral_flatness',   # White noise has higher flatness
            'spectral_entropy',    # Complexity of spectrum
            'spectral_centroid',   # Center of spectral mass
            'zcr',                 # Zero-crossing rate
        ]
        # Add ADEV features if available
        adev_cols = [c for c in df.columns if c.startswith('adev_')]
        if adev_cols:
            default_features.append(adev_cols[0])  # First ADEV tau
        
        features = [f for f in default_features if f in df.columns]
    
    n_features = min(len(features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    sensor_types = df[label_col].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sensor_types)))
    
    for idx, feat in enumerate(features[:n_features]):
        ax = axes[idx]
        
        for sensor, color in zip(sensor_types, colors):
            data = df[df[label_col] == sensor][feat].dropna()
            ax.hist(data, bins=30, alpha=0.5, label=sensor, color=color, density=True)
        
        ax.set_xlabel(feat)
        ax.set_ylabel('Density')
        ax.set_title(f'Distribution: {feat}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Feature Distributions by Sensor Type (Noise Characteristics)', fontsize=14)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('feature_distributions.png', dpi=150)
        print("Saved: feature_distributions.png")
    plt.close()


# =============================================================================
# 5. PROTOTYPE CLASSIFIER (IMPROVED)
# =============================================================================

def prototype_classifier(
    df: pd.DataFrame,
    label_col: str = 'sensor_type',
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[object, object, pd.DataFrame]:
    """
    Train a classifier on synthetic data to validate the pipeline.
    When real data arrives, you just swap the data source.
    
    Args:
        df: DataFrame with features and labels.
        label_col: Name of the label column.
        test_size: Fraction of data for testing.
        seed: Random seed.
        
    Returns:
        Tuple of (trained classifier, label_encoder, feature importances DataFrame).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
        return None, None, None
    
    print("=" * 60)
    print("PROTOTYPE CLASSIFIER")
    print("=" * 60)
    
    # Identify feature columns
    exclude_cols = METADATA_COLUMNS + [label_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Prepare features and labels
    X = df[feature_cols].fillna(0).values
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].values)
    
    print(f"\nDataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Classes: {list(le.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        random_state=seed, 
        n_jobs=-1
    )
    clf.fit(X_train_scaled, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Test set evaluation
    y_pred = clf.predict(X_test_scaled)
    test_accuracy = (y_pred == y_test).mean()
    
    print("\n" + "=" * 40)
    print("CLASSIFICATION REPORT")
    print("=" * 40)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nCONFUSION MATRIX")
    print("-" * 40)
    cm = confusion_matrix(y_test, y_pred)

    # Print confusion matrix using pandas
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print(cm_df.to_string())
    
    # Feature importance
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n" + "=" * 40)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 40)
    for _, row in importances.head(10).iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"{row['feature']:<25} {row['importance']:.4f} {bar}")
    
    # Store test accuracy for summary
    clf.test_accuracy_ = test_accuracy

    # Return dict with all data needed for visualization
    return {
        'clf': clf,
        'le': le,
        'importances': importances,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': clf.predict_proba(X_test_scaled),
        'classes': le.classes_,
        'test_accuracy': test_accuracy
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_full_analysis():
    """
    Run the complete analysis pipeline.
    """
    print("\n" + "=" * 70)
    print("PROJECT SHIELD - FULL ANALYSIS PIPELINE (v2)")
    print("=" * 70 + "\n")
    
    # 1. Physics Validation
    print("\n[1/5] PHYSICS VALIDATION")
    physics_results = validate_physics(save_plot=True)
    
    # 2. Sensor Modality Simulation (IMPROVED: zero_mean=True)
    print("\n\n[2/5] SENSOR MODALITY SIMULATION")
    sensor_df = simulate_sensor_modalities(
        n_samples=10000,
        fs=100.0,
        n_instances_per_class=5,
        zero_mean=True  # IMPROVED: Better for noise classification
    )
    
    # 3. Feature Discrimination
    print("\n\n[3/5] FEATURE DISCRIMINATION ANALYSIS")
    discrimination_df = analyze_feature_discrimination(sensor_df)
    
    # Visualize with IMPROVED default features
    visualize_feature_distributions(sensor_df)  # Uses good defaults now
    
    # 4. Degradation Analysis (IMPROVED: smoothed plots)
    print("\n\n[4/5] DEGRADATION ANALYSIS")
    degradation_results = analyze_all_degradation_modes(save_plot=True)
    
    # 5. Prototype Classifier
    print("\n\n[5/5] PROTOTYPE CLASSIFIER")
    clf_results = prototype_classifier(sensor_df)
    
    # Summary (FIXED accuracy calculation)
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - physics_validation.png")
    print("  - degradation_analysis.png")
    print("  - feature_distributions.png")
    print("\nKey findings:")
    print(f"  - Most discriminative feature: {discrimination_df.iloc[0]['feature']}")
    if clf_results is not None:
        print(f"  - Classifier test accuracy: {clf_results['test_accuracy']:.1%}")

    return {
        'physics': physics_results,
        'sensor_data': sensor_df,
        'discrimination': discrimination_df,
        'degradation': degradation_results,
        'classifier_results': clf_results,
    }


if __name__ == "__main__":
    # Run individual tests or full pipeline
    import sys
    
    if len(sys.argv) > 1:
        test_name = sys.argv[1]
        
        if test_name == 'physics':
            validate_physics()
        elif test_name == 'sensors':
            df = simulate_sensor_modalities()
            print(df.head())
        elif test_name == 'degradation':
            analyze_all_degradation_modes()
        elif test_name == 'discrimination':
            df = simulate_sensor_modalities()
            analyze_feature_discrimination(df)
        elif test_name == 'classifier':
            df = simulate_sensor_modalities()
            prototype_classifier(df)
        else:
            print(f"Unknown test: {test_name}")
            print("Available: physics, sensors, degradation, discrimination, classifier")
    else:
        # Run everything
        run_full_analysis()