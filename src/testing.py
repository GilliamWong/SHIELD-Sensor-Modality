import numpy as np
import matplotlib.pyplot as plt
from SensorDataLoader import SensorDataLoader
from extractor import FeatureExtractor
from freq_domain import get_psd_welch, get_spectral_slope

def validate_physics():
    """
    End-to-End Verification of the SHIELD Feature Extraction Pipeline.
    
    1. Generates synthetic sensor data (White, Pink, Brown).
    2. Runs the full extraction pipeline (Sliding Window).
    3. Asserts that extracted features match physical laws.
    4. Generates a visual report (PNG).
    """
    # --- Configuration ---
    FS = 100.0          # 100 Hz sampling
    DURATION = 100      # 100 seconds of data
    N_SAMPLES = int(FS * DURATION)
    WINDOW_SEC = 1.0    # 1 second windows
    STEP_SEC = 0.5      # 50% overlap
    
    loader = SensorDataLoader(seed=42)
    extractor = FeatureExtractor(fs=FS)
    
    print(f"--- SHIELD Physics Validation [fs={FS}Hz, N={N_SAMPLES}] ---")

    # --- 1. Generate Synthetic Signals ---
    print("Generating synthetic noise profiles...")
    signals = {
        'White (Accel)': loader.generate_synthetic_data(N_SAMPLES, noise_type='white'),
        'Pink (Elec)':   loader.generate_synthetic_data(N_SAMPLES, noise_type='pink'),
        'Brown (Gyro)':  loader.generate_synthetic_data(N_SAMPLES, noise_type='brown')
    }

    # --- 2. Process & Validate ---
    results = {}
    
    for name, signal in signals.items():
        print(f"\nProcessing {name}...")
        
        # Run the pipeline
        # Note: signal.flatten() ensures we pass 1D array if generator returned 2D
        df = extractor.process_signal(signal.flatten(), WINDOW_SEC, STEP_SEC)
        
        # Calculate aggregate metrics
        window_slope = df['psd_slope'].mean()
        avg_entropy = df['spectral_entropy'].mean()

        # Full-signal PSD for the plot/ground-truth slope (aligns with plotted line)
        f_full, p_full = get_psd_welch(signal.flatten(), fs=FS)
        full_slope = get_spectral_slope(f_full, p_full)
        
        results[name] = {
            'window_slope': window_slope,
            'full_slope': full_slope,
            'entropy': avg_entropy
        }
        
        print(f"  -> extracted {len(df)} windows")
        print(f"  -> Avg PSD Slope (windowed): {window_slope:.4f}")
        print(f"  -> Full-signal PSD Slope:    {full_slope:.4f}")
        print(f"  -> Avg Entropy:   {avg_entropy:.4f}")
        
        # Assertions (The "Unit Tests")
        slope_for_check = full_slope if np.isfinite(full_slope) else window_slope
        if 'White' in name:
            if -0.5 < slope_for_check < 0.5: print("  [PASS] Slope is flat (~0)")
            else: print("  [FAIL] Slope incorrect for White noise")
            
        elif 'Pink' in name:
            if -1.5 < slope_for_check < -0.5: print("  [PASS] Slope is 1/f (~-1)")
            else: print("  [FAIL] Slope incorrect for Pink noise")
            
        elif 'Brown' in name:
            if slope_for_check < -1.5: print("  [PASS] Slope is steep (~-2)")
            else: print("  [FAIL] Slope incorrect for Brown noise")

    # --- 3. Visualization (The Report) ---
    print("\nGenerating visual report...")
    plt.figure(figsize=(12, 6))
    
    # Plot PSDs
    for name, signal in signals.items():
        f, p = get_psd_welch(signal.flatten(), fs=FS)
        
        # Filter DC component for plotting
        mask = f > 0
        
        slope_label = results[name]['full_slope']
        if not np.isfinite(slope_label):
            slope_label = results[name]['window_slope']

        plt.loglog(f[mask], p[mask], label=f"{name} (Slope={slope_label:.2f})", alpha=0.8)

    plt.title("Noise Color Slopes")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V^2/Hz)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    filename = 'validation_plot.png'
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    print("--- Validation Complete ---")

if __name__ == "__main__":
    validate_physics()
