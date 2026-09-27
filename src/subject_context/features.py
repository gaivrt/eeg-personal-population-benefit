import numpy as np
from scipy.signal import butter, sosfiltfilt


def matrix_function(matrix, fn):
    vals, vecs = np.linalg.eigh(matrix)
    return (vecs * fn(np.maximum(vals, 1e-8))[..., None, :]) @ np.swapaxes(vecs, -1, -2)


def band_covariances(eeg, sfreq=200):
    values = []
    for low, high in [(8, 13), (13, 30)]:
        signal = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos"), eeg, axis=-1)
        signal -= signal.mean(axis=-1, keepdims=True)
        cov = signal @ np.swapaxes(signal, -1, -2) / (signal.shape[-1] - 1)
        # Fixed shrinkage, preserving SPD despite common average reference.
        trace = np.trace(cov, axis1=-2, axis2=-1) / cov.shape[-1]
        cov = .95 * cov + .05 * trace[..., None, None] * np.eye(cov.shape[-1])
        values.append(cov)
    return np.stack(values, axis=1)


class TangentSpace:
    """Log-Euclidean reference fit exclusively on training contexts."""
    def fit(self, covariances):
        reference = matrix_function(matrix_function(covariances, np.log).mean(axis=0), np.exp)
        self.whitener = matrix_function(reference, lambda v: v ** -0.5)
        return self

    def transform(self, covariances):
        logs = matrix_function(self.whitener @ covariances @ self.whitener, np.log)
        i, j = np.triu_indices(logs.shape[-1])
        flat = logs[..., i, j] * np.where(i == j, 1, np.sqrt(2))
        return flat.reshape(len(covariances), -1).astype(np.float32)
