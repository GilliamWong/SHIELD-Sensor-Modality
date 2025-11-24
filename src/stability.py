import numpy as np
import allantools
from typing import Dict

def get_allan_deviation(signal: np.ndarray, fs: float, taus: np.ndarray = None) -> Dict[str, float]:
    """
    Calculates Overlapping Allan Deviation (OADEV), the standard metric for 
    inertial sensor stability (gyroscopes/accelerometers).
    
    Physics Targets (Log-Log Slope):
        - Slope -0.5 = White Noise (Angle Random Walk)
        - Slope 0.0  = Flicker Noise (Bias Instability)
        - Slope +0.5 = Random Walk (Rate Random Walk)
        
    Args:
        signal: The sensor data array (1D).
        fs: Sampling frequency in Hz.
        taus: Specific integration times to check. If None, selects decades automatically.
        
    Returns:
        Dictionary containing ADEV values at specific taus and the estimated slope.
    """
    # default to decades if no taus provided
    if taus is None:
        # We can only go up to half the signal length
        max_tau = len(signal) / fs / 2.0
        # Create log-spaced taus: 0.1s, 1s, 10s, etc.
        taus = np.logspace(np.log10(1.0/fs), np.log10(max_tau), num=10)
        # Filter to ensure we don't exceed data length
        taus = taus[taus < max_tau]

    if len(taus) < 2:
        return {}

    try:
        # Calculate Overlapping Allan Deviation
        # data_type="freq" is appropriate for gyroscope rate data
        (taus_out, ad, ade, ns) = allantools.oadev(
            signal, rate=fs, data_type="freq", taus=taus
        )
    except Exception as e:
        # Fallback if signal is too short or calculation fails
        return {}

    features = {}
    
    # 1. Store specific tau values (e.g., ADEV at 1 second)
    for i, t in enumerate(taus_out):
        # Create clean key names like "adev_1.00s"
        key = f"adev_{t:.2f}s"
        features[key] = float(ad[i])
        
    # 2. Calculate ADEV Slope (The "Fingerprint")
    # We fit a line to log(tau) vs log(adev)
    if len(taus_out) > 2:
        log_t = np.log10(taus_out)
        log_ad = np.log10(ad)
        slope, _ = np.polyfit(log_t, log_ad, 1)
        features['adev_slope'] = float(slope)
        
    return features