"""Read EDF structural metadata without reading any signal data records."""
from math import isfinite


def parse_edf_header(header):
    if len(header) < 256 or header[:8].strip() != b"0":
        raise ValueError("Not a complete EDF fixed header")
    count = int(header[252:256])
    size = int(header[184:192])
    if not 1 <= count <= 512 or size != 256 * (count + 1) or len(header) != size:
        raise ValueError("Invalid EDF header length/channel count")
    records = int(header[236:244])
    record_seconds = float(header[244:252])
    if records < 0 or not isfinite(record_seconds) or record_seconds <= 0:
        raise ValueError("Unknown/invalid data record count or duration")
    labels = [header[256 + 16*i:256 + 16*(i+1)].decode("ascii").strip() for i in range(count)]
    offset = 256 + 216 * count
    samples = [int(header[offset + 8*i:offset + 8*(i+1)]) for i in range(count)]
    if any(n <= 0 for n in samples):
        raise ValueError("Non-positive samples per data record")
    eeg = [i for i, label in enumerate(labels) if label != "EDF Annotations"]
    return {"header_bytes": size, "signal_count": count, "eeg_channel_count": len(eeg),
            "eeg_labels": [labels[i] for i in eeg],
            "sfreqs": sorted(set(samples[i] / record_seconds for i in eeg)),
            "sample_counts": sorted(set(records * samples[i] for i in eeg)),
            "records": records, "record_seconds": record_seconds,
            "duration_seconds": records * record_seconds,
            "expected_file_bytes": size + 2 * records * sum(samples),
            "start_date_header": header[168:176].decode("ascii").strip(),
            "start_time_header": header[176:184].decode("ascii").strip()}
