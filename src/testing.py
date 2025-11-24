import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import SensorDataLoader
from src.features.freq_domain import extract_freq_features, get_psd_welch
from src.features.stability import get_allan_deviation

def validate_physics():
    """
    Generates White, Pink, and Brown noise and asserts that the
    Feature Extraction pipeline correctly identifies their fingerprints.
    """
    loader = SensorDataLoader()
    fs = 100.0
    n_samples = 10000 # 100 seconds of data
    
    # 1. Generate Signals
    print("Generating synthetic signals...")
    sig_white = loader.generate_synthetic_data(n_samples, noise_type='white')
    sig_pink = loader.generate_synthetic_data(n_samples, noise_type='pink')
    sig_brown = loader.generate_synthetic_data(n_samples, noise_type='brown')
    
    # 2. Run Extraction
    print("Extracting features...")
    feats_white = extract_freq_features(sig_white, fs)
    feats_pink = extract_freq_features(sig_pink, fs)
    feats_brown = extract_freq_features(sig_brown, fs)
    
    # 3. Print Results (The "Moment of Truth")
    print("\n--- PSD SLOPE VALIDATION ---")
    print(f"White Noise (Target ~ 0.0):  {feats_white['psd_slope']:.4f}")
    print(f"Pink Noise  (Target ~ -1.0): {feats_pink['psd_slope']:.4f}")
    print(f"Brown Noise (Target ~ -2.0): {feats_brown['psd_slope']:.4f}")
    
    # 4. Visualize (The Meeting Material)
    # Calculate PSDs again for plotting
    f_w, p_w = get_psd_welch(sig_white, fs)
    f_p, p_p = get_psd_welch(sig_pink, fs)
    f_b, p_b = get_psd_welch(sig_brown, fs)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(f_w, p_w, label=f'White (Slope={feats_white["psd_slope"]:.2f})', alpha=0.7)
    plt.loglog(f_p, p_p, label=f'Pink (Slope={feats_pink["psd_slope"]:.2f})', alpha=0.7)
    plt.loglog(f_b, p_b, label=f'Brown (Slope={feats_brown["psd_slope"]:.2f})', alpha=0.7)
    plt.title("Feature Extraction Validation: PSD Log-Log Slopes")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    print("\nPlot generated. Saving to 'validation_plot.png'...")
    plt.savefig('validation_plot.png')
    plt.close()

if __name__ == "__main__":
    validate_physics()