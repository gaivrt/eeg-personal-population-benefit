"""Resumeable audit of all 109 x 14 EDF headers, never download full signals."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_context.edf_header import parse_edf_header

LOCAL = threading.local()


def read_range(url, start, end, timeout):
    if not hasattr(LOCAL, "http"):
        LOCAL.http = requests.Session()
    with LOCAL.http.get(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                        stream=True, timeout=(10, timeout)) as response:
        response.raise_for_status()
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("Encoded response cannot be audited as raw EDF bytes")
        if response.status_code == 206:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if not match or (int(match[1]), int(match[2])) != (start, end):
                raise ValueError("Server returned a different byte range")
            total = int(match[3])
            body = response.raw.read(end - start + 1)
            read_bytes = len(body)
        elif response.status_code == 200:
            # Server ignored Range: stream only the prefix, close before reading signal.
            total = int(response.headers["Content-Length"])
            prefix = response.raw.read(end + 1)
            body, read_bytes = prefix[start:], len(prefix)
        else:
            raise ValueError(f"Unexpected HTTP status {response.status_code}")
        if len(body) != end - start + 1:
            raise ValueError("Truncated header range")
        return body, {"status": response.status_code, "content_range": response.headers.get("Content-Range"),
                      "body_bytes_read": read_bytes, "file_bytes": total,
                      "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")}


def inspect_one(subject, run, cfg, headers_dir):
    name = f"S{subject:03d}R{run:02d}"
    url = f'{cfg["base_url"]}/S{subject:03d}/{name}.edf'
    header_path, receipt_path = headers_dir / (name + ".header"), headers_dir / (name + ".json")
    if header_path.exists() and receipt_path.exists():
        header, receipt = header_path.read_bytes(), json.loads(receipt_path.read_text())
        if receipt["url"] != url or hashlib.sha256(header).hexdigest() != receipt["sha256"]:
            raise ValueError(f"Cached header checksum failed: {name}")
    else:
        for attempt in range(cfg["retries"]):
            try:
                first, r1 = read_range(url, 0, 255, cfg["timeout_seconds"])
                size = int(first[184:192])
                if not 512 <= size <= 131328 or size != 256 * (int(first[252:256]) + 1):
                    raise ValueError("Invalid header size")
                tail, r2 = read_range(url, 256, size - 1, cfg["timeout_seconds"])
                if r1["file_bytes"] != r2["file_bytes"] or r1["etag"] != r2["etag"]:
                    raise ValueError("EDF changed between header range reads")
                header = first + tail
                parse_edf_header(header)
                receipt = {"url": url, "utc": datetime.now(timezone.utc).isoformat(),
                           "sha256": hashlib.sha256(header).hexdigest(), "requests": [r1, r2],
                           "body_bytes_read": r1["body_bytes_read"] + r2["body_bytes_read"],
                           "file_bytes": r1["file_bytes"]}
                header_path.write_bytes(header)
                receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
                break
            except Exception:
                if attempt == cfg["retries"] - 1:
                    raise
                time.sleep(1 + attempt)
    info = parse_edf_header(header)
    return {"subject": subject, "run": run, "url": url, **info,
            "file_bytes": receipt["file_bytes"], "header_sha256": receipt["sha256"],
            "body_bytes_read": receipt["body_bytes_read"],
            "http_statuses": [r["status"] for r in receipt["requests"]]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_audit.yaml")
    args = parser.parse_args()
    cfg_all = yaml.safe_load((ROOT / args.config).read_text())
    cfg = cfg_all["physionet"]
    output, cache = ROOT / cfg_all["output"], ROOT / cfg["headers_dir"]
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    tasks = [(s, r) for s in range(cfg["subjects"][0], cfg["subjects"][1]+1)
             for r in range(cfg["runs"][0], cfg["runs"][1]+1)]
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = {pool.submit(inspect_one, s, r, cfg, cache): (s, r) for s, r in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({"subject": futures[future][0], "run": futures[future][1], "error": repr(exc)})
            if i % 100 == 0 or i == len(tasks):
                print(f"headers {i}/{len(tasks)}; failures={len(failures)}", flush=True)
    modes = {}
    for run in range(cfg["runs"][0], cfg["runs"][1]+1):
        counts = Counter(row["duration_seconds"] for row in rows if row["run"] == run)
        modes[run] = counts.most_common(1)[0][0] if counts else None
    for row in rows:
        flags = []
        if row["sfreqs"] != [cfg["expected_sfreq"]]:
            flags.append("sampling_rate")
        if row["duration_seconds"] != modes[row["run"]]:
            flags.append("duration_differs_from_run_mode")
        if row["eeg_channel_count"] != cfg["expected_eeg_channels"]:
            flags.append("channel_count")
        if row["file_bytes"] != row["expected_file_bytes"]:
            flags.append("file_size_mismatch")
        row["flags"] = flags
    rows.sort(key=lambda row: (row["subject"], row["run"]))
    anomalies = [r for r in rows if r["flags"]]
    for name, items in [("physionet_headers", rows), ("physionet_anomalies", anomalies)]:
        (output / f"{name}.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
        flat = [{k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items() if k != "eeg_labels"}
                for row in items]
        pd.DataFrame(flat).to_csv(output / f"{name}.csv", index=False)
    summary = {"expected_files": len(tasks), "audited_files": len(rows), "failures": failures,
               "duration_mode_by_run": modes, "flagged_subjects": sorted({r["subject"] for r in anomalies}),
               "body_bytes_read_successful_headers": sum(r["body_bytes_read"] for r in rows),
               "edf_file_bytes_represented": sum(r["file_bytes"] for r in rows),
               "header_only": True, "duration_flag_is_descriptive_not_exclusion": True}
    (output / "physionet_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit("Some headers failed; rerun to resume missing files")


if __name__ == "__main__":
    main()
