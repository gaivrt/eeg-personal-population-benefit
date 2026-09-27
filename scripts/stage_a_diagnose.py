"""Read-only amplitude diagnosis for the already processed Cho sample."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import provenance, require_compute, scratch_root, save_json, sha256

require_compute()
root = scratch_root()
out = root / "processed/Cho2017/sub-001"
metadata = json.loads((out / "metadata.json").read_text())
source = loadmat(metadata["source_files"][0]["path"], simplify_cells=True)["eeg"]
diagnosis = {"provenance": provenance(), "diagnostic_source_sha256": sha256(__file__)}
for name in ["rest", "imagery_left", "imagery_right"]:
    x = source[name][:64].astype(np.float64)
    diagnosis[name] = {"shape": list(x.shape), "range_raw_units": [float(x.min()), float(x.max())],
                       "channel_ptp_quantiles": np.quantile(np.ptp(x, axis=-1), [0, .5, .95, 1]).tolist(),
                       "median_absolute_channel_mean": float(np.median(np.abs(x.mean(axis=-1))))}
for name in ["rest_open", "tasks"]:
    x = np.load(out / f"{name}.npy").astype(np.float64) * 100
    ptp = np.ptp(x, axis=-1)
    diagnosis[name] = {"ptp_uv_quantiles": np.quantile(ptp, [0, .5, .95, 1]).tolist(),
                       "channels_with_max_ptp": [metadata["channels"][i] for i in np.argsort(ptp.max(axis=0))[-5:]],
                       "flat_windows": int(np.any(ptp < .01, axis=1).sum()),
                       "over_1000_windows": int(np.any(ptp > 1000, axis=1).sum())}
save_json(root / "receipts/cho_amplitude_diagnosis.json", diagnosis)
print(json.dumps(diagnosis, indent=2))
