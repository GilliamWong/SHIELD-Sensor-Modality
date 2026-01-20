import numpy as np
from scipy.stats import kurtosis, skew
from typing import Iterable, Mapping

#calculates the mean, variance, RMS, skewness, kurtosis, optional quantiles, and zero-crossing rate.
def get_statistical_moments(signal: np.ndarray, quantiles: Iterable[float] = (0.25, 0.5, 0.75)) -> Mapping[str, float]:
    values = np.asarray(signal, dtype=np.float64).flatten()
    if values.size == 0:
        return {}

    mean_val = float(np.mean(values))
    variance = float(np.var(values))
    rms = float(np.sqrt(np.mean(values ** 2)))
    skewness = float(skew(values, bias=False))
    kurt = float(kurtosis(values, bias=False))

    features = {
        'mean': mean_val,
        'variance': variance,
        'rms': rms,
        'skewness': skewness,
        'kurtosis': kurt,
    }

    # zero crossing rate
    signs = np.sign(values)
    zero_crossings = np.count_nonzero(np.diff(signs))
    features['zcr'] = zero_crossings / max(len(values) - 1, 1)

    if quantiles is not None:
        for q in quantiles:
            key = f"quantile_{int(q * 100)}"
            features[key] = float(np.quantile(values, q))

    return features
