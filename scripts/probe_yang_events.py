"""Read only S001's three event files and BDF headers from the official ZIP."""
import hashlib
import json
import zipfile

from inspect_rest_archives import RangeFile, OUT


if __name__ == "__main__":
    stream = RangeFile("https://ndownloader.figshare.com/files/51001884", 65630556299, "yang")
    receipts = []
    try:
        with zipfile.ZipFile(stream) as z:
            for session in (1, 2, 3):
                base = f"WBCIC_SHU Motor Imagery dataset/sourcedata/2C dataset/sub-001/ses-{session:02d}/eeg/"
                folder = OUT / "yang_s001" / f"ses-{session:02d}"
                folder.mkdir(parents=True, exist_ok=True)
                events = z.read(base + "evt.bdf")
                (folder / "evt.bdf").write_bytes(events)
                with z.open(base + "data.bdf") as f:
                    fixed = f.read(256)
                    size = int(fixed[184:192])
                    header = fixed + f.read(size-256)
                (folder / "data.header").write_bytes(header)
                receipts.append({"session": session, "source_prefix": base,
                                 "evt_sha256": hashlib.sha256(events).hexdigest(),
                                 "header_sha256": hashlib.sha256(header).hexdigest(),
                                 "evt_bytes": len(events), "header_bytes": len(header)})
                print(receipts[-1], flush=True)
    finally:
        (OUT / "yang_event_probe_receipts.json").write_text(json.dumps(
            {"members": receipts, "range_bytes_requested": stream.used, "ranges": stream.receipts}, indent=2))
