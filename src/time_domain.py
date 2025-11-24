import numpy as np
from scipy.stats import skew, kurtosis

#calculates the mean, variance, skewness, and kurtosis of a signal.
#Returns: dict: {'mean': float, 'var': float, 'skew': float, 'kurt': float}
def get_statistical_moments(signal: np.ndarray) -> dict:
    mean = np.mean(signal)
    var = np.var(signal)
    skew = skew(signal)
    kurt = kurtosis(signal)
    return {'mean': mean, 'var': var, 'skew': skew, 'kurt': kurt}