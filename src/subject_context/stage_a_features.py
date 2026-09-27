"""EA and covariance features from the saved float16 signals only."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

def covariances(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)
    c = x @ x.swapaxes(-1, -2) / (x.shape[-1] - 1)
    scale = np.trace(c, axis1=-2, axis2=-1) / c.shape[-1]
    return .999 * c + .001 * np.maximum(scale, 1e-12)[..., None, None] * np.eye(c.shape[-1])

def inverse_sqrt(c):
    values, vectors = np.linalg.eigh(c)
    return (vectors * np.maximum(values, np.max(values) * 1e-8) ** -.5) @ vectors.T

def ea_references(signals):
    covs = {k: covariances(v) for k, v in signals.items()}
    return {"all": inverse_sqrt(np.concatenate(list(covs.values())).mean(0)),
            "eo": inverse_sqrt(covs["open"].mean(0)),
            "task": inverse_sqrt(covs["tasks"].mean(0))}

def classic_covariances(signals, picks):
    sos = butter(4, [8, 30], fs=200, btype="bandpass", output="sos")
    filtered = {k: sosfiltfilt(sos, np.asarray(v[:, picks], dtype=np.float64), axis=-1) for k,v in signals.items()}
    refs = ea_references(filtered)
    covs = covariances(filtered["tasks"])
    return {k: w @ covs @ w.T for k,w in refs.items() if k in ("all", "eo")}
