"""Download the fixed starting cohort via MOABB, exclusively into scratch."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from subject_context.stage_a_common import (
    participants, provenance, require_compute, save_json, scratch_root, sha256,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["PhysionetMI", "Dreyer2023", "Cho2017"])
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    workers = args.workers or (1 if args.dataset == "Dreyer2023" else 3)
    require_compute()
    root = scratch_root()
    raw_root = root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    for key in ["MNE_DATA", "MNE_DATASETS_EEGBCI_PATH", "MNE_DATASETS_GIGADB_PATH",
                "MNE_DATASETS_DREYER2023_PATH"]:
        os.environ[key] = str(raw_root)
    os.environ["_MNE_FAKE_HOME_DIR"] = str(root / "mne-config")
    from moabb.datasets import Cho2017, Dreyer2023, PhysionetMI
    from moabb.datasets import download as moabb_download
    # The locked MOABB build defaults verify=False. Set it explicitly before
    # MOABB's setdefault call; preserve its URL/path/reader logic.
    original_choose_downloader = moabb_download.choose_downloader
    def verified_downloader(url, **kwargs):
        kwargs["progressbar"] = False
        downloader = original_choose_downloader(url, **kwargs)
        if hasattr(downloader, "kwargs"):
            downloader.kwargs["verify"] = True
            downloader.kwargs["timeout"] = 120
        if url.startswith("https://physionet.org/files/eegmmidb/1.0.0/"):
            # PhysioNet itself documents this public S3 mirror for bulk access.
            # Keep MOABB's original cache path so previous files are reused.
            mirror = url.replace("https://physionet.org/files/", "https://physionet-open.s3.amazonaws.com/", 1)
            def download_mirror(requested_url, output_file, pooch, check_only=False):
                return downloader(mirror, output_file, pooch, check_only=check_only)
            download_mirror.kwargs = downloader.kwargs
            return download_mirror
        return downloader
    moabb_download.choose_downloader = verified_downloader
    import mne
    mne.set_log_level("WARNING")
    dataset = {"PhysionetMI": PhysionetMI, "Dreyer2023": Dreyer2023, "Cho2017": Cho2017}[args.dataset]()
    if args.dataset == "PhysionetMI":
        dataset.feet_runs = []
        dataset.hand_runs = [4, 8, 12]
    out = root / "receipts/download" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    receipt = provenance()
    if (out / "run.json").exists():
        previous = json.loads((out / "run.json").read_text())
        save_json(out / ("attempt-" + previous["slurm_job_id"] + ".json"), previous)
    receipt.update(dataset=args.dataset, status="running", subjects=participants(args.dataset), files=[],
                   tls_verify=True, workers=workers,
                   physionet_transport="official physionet-open S3 mirror; original MOABB cache paths")
    save_json(out / "run.json", receipt)
    def download_one(subject):
            subject_start = time.monotonic()
            print(json.dumps({"dataset": args.dataset, "subject": subject, "status": "starting"}), flush=True)
            if args.dataset == "Dreyer2023":
                path = Path(dataset.download_by_subject(subject, path=str(raw_root)))
                files = sorted((path / f"sub-{subject:02d}").rglob("*"))
                archive = path / f"sub-{subject:02d}.zip"
                if archive.exists():
                    files.append(archive)
            else:
                paths = dataset.data_path(subject, path=str(raw_root), update_path=False, verbose=False)
                files = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
            records = []
            for path in files:
                if not path.is_file():
                    continue
                if not path.resolve().is_relative_to(raw_root.resolve()):
                    raise RuntimeError(f"Download escaped scratch: {path}")
                records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
            row = {"subject": subject, "seconds": time.monotonic() - subject_start,
                   "files": records, "bytes_on_disk": sum(r["bytes"] for r in records)}
            save_json(out / f"subject-{subject:03d}.json", row)
            print(json.dumps({"dataset": args.dataset, "subject": subject,
                              "seconds": row["seconds"], "bytes": row["bytes_on_disk"], "status": "complete"}), flush=True)
            return records
    def download_subject(subject):
        import requests
        for attempt in range(5):
            try:
                return download_one(subject)
            except requests.exceptions.HTTPError as exc:
                if exc.response is None or exc.response.status_code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                retry_after = exc.response.headers.get("Retry-After", "")
                delay = max(60 * (2 ** attempt), int(retry_after) if retry_after.isdigit() else 0)
                print(json.dumps({"subject": subject, "http_status": exc.response.status_code,
                                  "retry_after_seconds": delay}), flush=True)
                time.sleep(delay)
    try:
        subjects = participants(args.dataset)
        # Initialize the shared Dreyer manifest once before concurrent subjects.
        if args.dataset == "Dreyer2023":
            receipt["files"].extend(download_subject(subjects.pop(0)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_subject, s): s for s in subjects}
            for future in as_completed(futures):
                receipt["files"].extend(future.result())
                save_json(out / "run.json", receipt)
        receipt["status"] = "complete"
    except Exception:
        receipt["status"] = "failed"
        receipt["error"] = traceback.format_exc()
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - begin
        unique = {f["path"]: f for f in receipt["files"]}
        receipt["bytes_on_disk"] = sum(f["bytes"] for f in unique.values())
        # ZIP + extracted content are both retained; do not call their sum network volume.
        receipt["downloaded_payload_bytes"] = sum(f["bytes"] for f in unique.values()
            if args.dataset != "Dreyer2023" or f["path"].endswith(".zip"))
        save_json(out / "run.json", receipt)


if __name__ == "__main__":
    main()
