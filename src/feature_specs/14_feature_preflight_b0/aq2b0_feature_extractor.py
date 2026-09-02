import argparse
import csv
import hashlib
import json
import wave
from pathlib import Path

import numpy as np


REPRESENTATION_A_ID = "STFT_LOGPOWER_STATS_V1"
REPRESENTATION_B_ID = "STFT_LOGPOWER_TENSOR_V1"


def read_pcm16_wav(path):
    with wave.open(str(path), "rb") as w:
        sample_rate = w.getframerate()
        channels = w.getnchannels()
        sample_width = w.getsampwidth()
        sample_count = w.getnframes()
        frames = w.readframes(sample_count)
    if sample_rate != 48000:
        raise ValueError(f"invalid sample_rate={sample_rate}")
    if channels != 1:
        raise ValueError(f"invalid channels={channels}")
    if sample_width != 2:
        raise ValueError(f"invalid sample_width={sample_width}")
    if sample_count != 96000:
        raise ValueError(f"invalid sample_count={sample_count}")
    x_int16 = np.frombuffer(frames, dtype="<i2")
    return x_int16.astype(np.float64) / 32768.0


def symmetric_hann(win_length):
    n = np.arange(win_length, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(win_length - 1))


def extract_stft_core(x, spec):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (spec["sample_count"],):
        raise ValueError(f"invalid input shape={x.shape}")
    n_fft = int(spec["n_fft"])
    win_length = int(spec["win_length"])
    hop = int(spec["hop_length"])
    n_frames = int(spec["n_frames"])
    window = symmetric_hann(win_length)
    win_power = float(np.sum(window * window))
    edges = np.floor(np.linspace(0, int(spec["fft_bins"]), int(spec["frequency_groups"]) + 1)).astype(np.int64)
    if edges[0] != 0 or edges[-1] != int(spec["fft_bins"]) or np.any(np.diff(edges) < 1):
        raise ValueError("invalid frequency group edges")
    grouped = np.empty((int(spec["frequency_groups"]), n_frames), dtype=np.float64)
    eps = float(spec["epsilon"])
    for t in range(n_frames):
        start = t * hop
        frame = x[start:start + win_length]
        if frame.shape[0] != win_length:
            raise ValueError("incomplete frame")
        spectrum = np.fft.rfft(frame * window, n=n_fft)
        power = (np.abs(spectrum) ** 2) / win_power
        log_power = 10.0 * np.log10(np.maximum(power, eps))
        for b in range(int(spec["frequency_groups"])):
            grouped[b, t] = np.mean(log_power[edges[b]:edges[b + 1]])
    if grouped.shape != tuple(spec["representation_B_shape"]):
        raise ValueError(f"invalid grouped shape={grouped.shape}")
    return grouped


def extract_representation_a(G, spec):
    G = np.asarray(G, dtype=np.float64)
    temporal_mean = np.mean(G, axis=1)
    temporal_std = np.std(G, axis=1, ddof=0)
    feature = np.concatenate([temporal_mean, temporal_std]).astype(np.float64, copy=False)
    if feature.shape != (int(spec["representation_A_dimension"]),):
        raise ValueError(f"invalid representation A shape={feature.shape}")
    if not np.all(np.isfinite(feature)):
        raise ValueError("non-finite representation A")
    return feature


def extract_representation_b(G, spec):
    tensor = np.asarray(G, dtype=np.float64)
    if tensor.shape != tuple(spec["representation_B_shape"]):
        raise ValueError(f"invalid representation B shape={tensor.shape}")
    if not np.all(np.isfinite(tensor)):
        raise ValueError("non-finite representation B")
    return tensor


def canonical_array_hash(arr, representation_id):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    shape = ",".join(str(x) for x in arr.shape)
    payload = (
        representation_id.encode("ascii")
        + b"\n"
        + shape.encode("ascii")
        + b"\n"
        + b"<f8"
        + b"\n"
        + arr.tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def process_inventory(input_inventory, spec_path, rep_a_dir, rep_b_dir, manifest_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rep_a_dir = Path(rep_a_dir)
    rep_b_dir = Path(rep_b_dir)
    rep_a_dir.mkdir(parents=True, exist_ok=True)
    rep_b_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(input_inventory, newline="", encoding="utf-8") as f:
        inputs = sorted(csv.DictReader(f), key=lambda r: r["sample_uid"])
    for item in inputs:
        x = read_pcm16_wav(item["pcm_path"])
        G = extract_stft_core(x, spec)
        arr_a = extract_representation_a(G, spec)
        arr_b = extract_representation_b(G, spec)
        outputs = [
            (REPRESENTATION_A_ID, arr_a, rep_a_dir / f"{item['sample_uid']}.npy"),
            (REPRESENTATION_B_ID, arr_b, rep_b_dir / f"{item['sample_uid']}.npy"),
        ]
        for representation_id, arr, path in outputs:
            np.save(path, arr, allow_pickle=False)
            rows.append({
                "sample_uid": item["sample_uid"],
                "pcm_sha256": item["pcm_sha256"],
                "representation_id": representation_id,
                "shape": ",".join(str(x) for x in arr.shape),
                "dtype": str(np.asarray(arr).dtype),
                "finite_check": bool(np.all(np.isfinite(arr))),
                "array_sha256": canonical_array_hash(arr, representation_id),
                "npy_file_sha256": file_sha256(path),
                "npy_path": str(path),
            })
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        fields = ["sample_uid", "pcm_sha256", "representation_id", "shape", "dtype", "finite_check", "array_sha256", "npy_file_sha256", "npy_path"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-inventory", required=True)
    parser.add_argument("--feature-spec", required=True)
    parser.add_argument("--representation-a-dir", required=True)
    parser.add_argument("--representation-b-dir", required=True)
    parser.add_argument("--feature-manifest", required=True)
    args = parser.parse_args()
    process_inventory(args.input_inventory, args.feature_spec, args.representation_a_dir, args.representation_b_dir, args.feature_manifest)


if __name__ == "__main__":
    main()
