"""Transparent storage and hypothetical GPU-throughput scenarios, no GPU use."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs/data_access_audit_2026-09-25"


def giga_bytes(filename, predicate):
    tree = ET.fromstring((AUDIT / "sources" / filename).read_text())
    return sum(int(c.find("{*}Size").text) for c in tree.findall("{*}Contents") if predicate(c.find("{*}Key").text))


raw = {
    "Stieger2021_catalog_all_598_sessions": json.loads((AUDIT / "stieger_size_summary.json").read_text())["total_bytes"],
    "Lee2019_MI_catalog": giga_bytes("lee_object_list.xml", lambda k: k.endswith("_EEG_MI.mat")),
    "Cho2017_catalog": giga_bytes("cho_object_list.xml", lambda k: k.endswith(".mat")),
    "Dreyer2023_subject_archives_catalog": json.loads((AUDIT / "dreyer_size_summary.json").read_text())["total_archive_bytes"],
    "PhysionetMI_LR_plus_baselines_estimate": 109 * (2 * 1275936 + 3 * 2596896),
}
# Trial counts are budget scenarios, before artifact / duration exclusions.
trials = {"PhysionetMI": (109 * 45, 64), "Stieger2021": (598 * 225, 60),
          "Lee2019_MI": (54 * 2 * 200, 62), "Cho2017": (52 * 200, 64), "Dreyer2023": (87 * 240 - 80, 27)}
prefix_fp32 = {name: n * c * 4 * 200 * 4 for name, (n, c) in trials.items()}
samples = json.loads((ROOT / "reports/stage0a/download_receipts.json").read_text())
directories = {}
for name in ["data/samples", "weights", "vendor", "artifacts/stage0a", ".venv", ".cache"]:
    directories[name] = sum(p.stat().st_size for p in (ROOT / name).rglob("*") if p.is_file())
result = {"units": "bytes; GB means decimal 10**9", "raw_catalog_or_estimate": raw,
          "raw_total_bytes": sum(raw.values()), "downloaded_stage0a_bytes_including_weights": sum(r["bytes"] for r in samples),
          "logical_directory_bytes_may_double_count_hardlinks": directories,
          "free_disk_bytes": shutil.disk_usage(ROOT).free,
          "query_count_scenario_before_exclusions": sum(n for n, c in trials.values()),
          "query_prefix_fp32_bytes_by_dataset": prefix_fp32,
          "query_prefix_fp32_bytes_total": sum(prefix_fp32.values()),
          "query_prefix_fp16_bytes_total": sum(prefix_fp32.values()) // 2,
          "gpu_measured": False,
          "gpu_cache_hours_scenario": {str(speed): (sum(n for n, c in trials.values()) + 954 * 150) / speed / 3600 for speed in [30, 300]},
          "gpu_train_hours_per_condition_scenario": {str(speed): 25 * 2000 * 32 / speed / 3600 for speed in [200, 1000]},
          "gpu_scenario_assumptions": "Hypothetical full-backbone 30–300 windows/s; last-two-layer training 200–1000 query examples/s; 5 folds x 5 seeds x 2000 steps x batch 32. These are not measured throughput or a selected training budget."}
(ROOT / "reports/stage0a/resources.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
