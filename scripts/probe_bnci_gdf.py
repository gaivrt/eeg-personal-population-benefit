"""Extract two bounded original GDF members for calibration-marker auditing."""
import hashlib
import json
import zipfile
import zlib

from inspect_rest_archives import ROOT, OUT, RangeFile


if __name__ == "__main__":
    stream = RangeFile("https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip",
                       439968864, "bnci", budget=65_000_000)
    receipts = []
    try:
        with zipfile.ZipFile(stream) as z:
            members = [{"name": f.filename, "bytes": f.file_size, "compressed": f.compress_size}
                       for f in z.infolist()]
            (OUT / "bnci_gdf_members.json").write_text(json.dumps(members, indent=2))
            for name in ("A01T.gdf", "A04T.gdf"):
                matches = [f for f in z.infolist() if f.filename.rsplit("/", 1)[-1] == name]
                if len(matches) != 1: raise ValueError("Ambiguous GDF member")
                item = matches[0]
                if item.file_size > 50_000_000: raise ValueError("Unexpected GDF size")
                path = ROOT / "data/external-test/BNCI2014_001/original_gdf" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                print(name, item.file_size, "bytes, compressed", item.compress_size, flush=True)
                if not path.exists():
                    with z.open(item) as source, path.open("wb") as target:
                        while chunk := source.read(1024*1024): target.write(chunk)
                crc = 0
                with path.open("rb") as f:
                    while chunk := f.read(1024*1024): crc = zlib.crc32(chunk, crc)
                if path.stat().st_size != item.file_size or crc != item.CRC:
                    raise ValueError("Local GDF disagrees with source ZIP size/CRC")
                with path.open("rb") as f: sha = hashlib.file_digest(f, "sha256").hexdigest()
                receipts.append({"member": item.filename, "bytes": path.stat().st_size,
                                 "path": str(path.relative_to(ROOT)), "sha256": sha})
                print("verified ZIP CRC; SHA256", sha, flush=True)
    finally:
        (OUT / "bnci_gdf_receipts.json").write_text(json.dumps(
            {"members": receipts, "range_bytes_requested": stream.used, "ranges": stream.receipts}, indent=2))
