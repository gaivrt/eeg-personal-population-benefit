"""Bounded public-source retrieval for the outstanding resting-state audit."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/data_audit/followup_sources"
SOURCES = {
    "stieger_readme.pdf": "https://ndownloader.figshare.com/files/25302482",
    "stieger_original_study.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7727383/fullTextXML",
    "stieger_descriptor.xml": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8016873/fullTextXML",
    "lee_rest_study.html": "https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2020.00321/full",
    "yang_events.json": "https://ndownloader.figshare.com/files/44074955",
    "yang_eeg.json": "https://ndownloader.figshare.com/files/44074949",
    "yang_code.rar": "https://ndownloader.figshare.com/files/44074934",
}


def fetch(item):
    name, url = item
    path = OUT / name
    if not path.exists():
        with requests.get(url, stream=True, timeout=(10, 45)) as response:
            response.raise_for_status()
            chunks, total = [], 0
            for chunk in response.iter_content(65536):
                total += len(chunk)
                if total > 5_000_000:
                    raise ValueError(f"Source exceeds 5 MB metadata budget: {name}")
                chunks.append(chunk)
        path.write_bytes(b"".join(chunks))
    data = path.read_bytes()
    return {"name": name, "url": url, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    receipts = []
    with ThreadPoolExecutor(4) as pool:
        for item, future in [(item, pool.submit(fetch, item)) for item in SOURCES.items()]:
            try:
                result = future.result()
            except Exception as exc:
                result = {"name": item[0], "url": item[1], "error": repr(exc)}
            receipts.append(result)
            print(result, flush=True)
    (OUT / "receipts.json").write_text(json.dumps(receipts, indent=2), encoding="utf-8")
