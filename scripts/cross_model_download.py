"""Fetch only the pinned public pretrained weights; never execute a model or submit jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from subject_context.stage_a_common import save_json, sha256


def fetch(url, path, size, expected_sha256=None, git_blob_sha1=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with tmp.open("wb") as stream:
                for block in response.iter_content(1 << 20):
                    stream.write(block)
        if tmp.stat().st_size != size:
            raise ValueError(f"Size mismatch: {path}")
        tmp.replace(path)
    digest = sha256(path)
    if path.stat().st_size != size or (expected_sha256 and digest != expected_sha256):
        raise ValueError(f"Weight identity mismatch: {path}")
    if git_blob_sha1:
        blob = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                blob.update(block)
        if blob.hexdigest() != git_blob_sha1:
            raise ValueError(f"Git blob mismatch: {path}")
    return {"path": str(path), "url": url, "bytes": size, "sha256": digest,
            "published_sha256": expected_sha256, "git_blob_sha1": git_blob_sha1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "weights/cross_model")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/cross_model_x.yaml").read_text(encoding="utf-8"))
    reve, labram = (cfg["models"][m] for m in ("REVE", "LaBraM"))
    records = []
    records.append(fetch(f'https://huggingface.co/{reve["weights_repository"]}/resolve/{reve["weights_revision"]}/{reve["weights_filename"]}',
                         args.out / "REVE/model.safetensors", reve["bytes"], reve["published_sha256"]))
    print("REVE identity verified", flush=True)
    revision = reve["positions_revision"]
    tree_url = f"https://huggingface.co/api/models/brain-bzh/reve-positions/tree/{revision}"
    response = requests.get(tree_url, timeout=30)
    response.raise_for_status()
    entries = [f for f in response.json() if f["path"] == "model.safetensors"]
    if len(entries) != 1 or "lfs" not in entries[0]:
        raise ValueError("Pinned position weights missing from upstream manifest")
    entry = entries[0]
    records.append(fetch(f"https://huggingface.co/brain-bzh/reve-positions/resolve/{revision}/model.safetensors",
                         args.out / "REVE/positions.safetensors", entry["size"], entry["lfs"]["oid"]))
    records.append(fetch(f'https://raw.githubusercontent.com/935963004/LaBraM/{labram["code_revision"]}/{labram["weights_path"]}',
                         args.out / "LaBraM/labram-base.pth", labram["bytes"], git_blob_sha1=labram["git_blob_sha1"]))
    receipt = {"files": records, "config_sha256": sha256(ROOT / "configs/cross_model_x.yaml"),
               "GPU_jobs": 0, "model_executed": False}
    save_json(args.out / "manifest.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
