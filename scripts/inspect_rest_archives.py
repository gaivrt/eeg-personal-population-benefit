"""Read official archive directories/small metadata using bounded HTTP Range."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/data_audit/followup_sources"


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, name, budget=12_000_000):
        self.url, self.size, self.name, self.budget = url, size, name, budget
        self.pos, self.used = 0, 0
        self.receipts = []
        self.http = requests.Session()
        self.etag = None

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        if not 0 <= self.pos <= self.size: raise ValueError("Out of archive bounds")
        return self.pos

    def read(self, size=-1):
        if size < 0: size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if not size: return b""
        if self.used + size > self.budget or size > 5_000_000:
            raise ValueError("Archive probe exceeded metadata byte budget")
        start, end = self.pos, self.pos+size-1
        key = f"{self.name}_{start}_{end}.range"
        cache = OUT / "ranges" / key
        if cache.exists():
            body = cache.read_bytes()
            receipt = json.loads(cache.with_suffix(".json").read_text())
            if hashlib.sha256(body).hexdigest() != receipt["sha256"]:
                raise ValueError("Cached range hash mismatch")
            if (receipt["url"], receipt["start"], receipt["end"], receipt["total"]) != (self.url, start, end, self.size):
                raise ValueError("Cached range source mismatch")
        else:
            with self.http.get(self.url, headers={"Range": f"bytes={start}-{end}",
                                                  "Accept-Encoding": "identity"},
                               stream=True, timeout=(10, 45)) as response:
                if response.status_code != 206:
                    raise ValueError(f"Range not honored: HTTP {response.status_code}; full archive was not read")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if not match or tuple(map(int, match.groups())) != (start, end, self.size):
                    raise ValueError("Wrong archive Content-Range")
                body = response.raw.read(size)
                receipt = {"url": self.url, "start": start, "end": end, "total": self.size,
                           "status": response.status_code, "etag": response.headers.get("ETag"),
                           "sha256": hashlib.sha256(body).hexdigest()}
            if len(body) != size: raise ValueError("Truncated archive range")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(body)
            cache.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
        if len(body) != size: raise ValueError("Truncated archive range")
        if self.etag is not None and self.etag != receipt["etag"]:
            raise ValueError("Archive changed between range requests")
        self.etag = receipt["etag"]
        self.receipts.append(receipt)
        self.used += size
        self.pos += size
        return body


def inspect(name, url, size):
    stream = RangeFile(url, size, name)
    try:
        with zipfile.ZipFile(stream) as archive:
            members = [{"name": f.filename, "bytes": f.file_size, "compressed_bytes": f.compress_size,
                        "offset": f.header_offset, "compression": f.compress_type} for f in archive.infolist()]
            (OUT / f"{name}_members.json").write_text(json.dumps(members, indent=2, ensure_ascii=False), encoding="utf-8")
            print(name, "members", len(members), "directory bytes read", stream.used, flush=True)
            if name == "yang":
                for row in members[:35]: print(row, flush=True)
            else:
                for row in members:
                    path = row["name"]
                    if path.lower().endswith((".m", ".txt", ".md")) and any(
                            word in path.lower() for word in ["rest", "eyesopen", "paradigm", "raw2", "convert"]):
                        print(path, row["bytes"], flush=True)
                        if row["bytes"] < 100_000 and "eeglab" not in path.lower():
                            target = OUT / "openbmi_release_code" / path
                            if not target.resolve().is_relative_to((OUT / "openbmi_release_code").resolve()):
                                raise ValueError("Archive member escapes extraction directory")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(archive.read(path))
    finally:
        (OUT / f"{name}_range_receipts.json").write_text(json.dumps(stream.receipts, indent=2), encoding="utf-8")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True, parents=True)
    inspect("yang", "https://ndownloader.figshare.com/files/51001884", 65630556299)
    inspect("openbmi", "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/OpenBMI-master.zip", 112786143)
