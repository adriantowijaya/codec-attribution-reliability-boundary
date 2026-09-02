import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import wave
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import correlate
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
import sklearn


ROOT = Path(r"[REDACTED_LOCAL_PATH]\Tony Hidayat\AQ_OpenWorld_Codec_Provenance")
MM_ROOT = Path(r"[REDACTED_LOCAL_PATH]\Tony Hidayat\MM_Audio_Routing")
PYTHON = Path(r"[REDACTED_LOCAL_PATH]\anaconda3\python.exe")
CLASS_ORDER = ["AAC", "MP3", "Opus"]
CORPORA = ["Orchset", "IRMAS", "FSDnoisy18k"]
RATES = [32, 48, 64, 80, 96, 128]
MID_RATES = [48, 64, 80, 96]
NEW_RATES = [48, 80, 96]
SEGMENT_SAMPLES = 96000
SEGMENT_SECONDS = 2.0
ALIGN_BOUND = 8192
REP_ID = "LOGSPECTRAL_TEMPORAL_MODULATION_V1"
EXPECTED = {
    "AQ2C1B_FREEZE": "c7dfd6a06e32616fd9396e827af1f170e90f421c2b1f6d2a2997f3aa0fd86579",
    "AQ2C1B_PREPREDICTION": "a9a19c65abbeed2dd8bcd7c1fdd8815511f10baa384a192d8132429c5e290c79",
    "C1B_SPEC": "cc65c5b616e03106dbccbadeb2be1a36bf0f66ecdd67700dc7e94ac3f9cf1a5f",
    "DISCOVERY_EXCLUSION": "0ee6c0cfb62bfcfd98871af6e1d486e06caa4d578dab723abfec03a4f94e3395",
    "C1B_PARENT_MANIFEST": "8f5cace862a0066e7bc6c47e23816206dc9069d8841234186c20a747868e69d8",
    "C1A_CANDIDATE_INVENTORY": "bc222add63b3c7b69b574dfa05f4d016e5378522a6eac5b5e2956ad69dc28ed4",
    "C1B_SUMMARY": "7481021151676ff084f2562addedf765a752625b277fd95d3a1ad6825a12b275",
    "B1_DERIVATIVE_MANIFEST": "6f32143b4e0267e3f2acdf91404954a9ddb1e2ad5c5d5bf6cb99651dc9e021aa",
    "PYTHON": "62c225fb9cdc41b139c7024581c233644f975ffc35314558c60ebefa6b88be01",
    "ENCODER": "57c56e369d5b4873b4d93fc1a1d833cb7cd8bc9325c14b05c34ce60b22842d8a",
    "FFPROBE": "afe05347caaabe479b3c4eae71992b6ec1e11c57266a1d665deb0f9fe9847208",
    "DECODER": "4dc3e63209cb6f183b703c8842f6e3dcc22778ccca1a3b9f4b5fca4034bb54dd",
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def raw_pcm_sha256(path):
    with wave.open(str(path), "rb") as w:
        return hashlib.sha256(w.readframes(w.getnframes())).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_cmd(cmd, timeout=120):
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return {"command": " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), "return_code": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}


def wav_read(path):
    with wave.open(str(path), "rb") as w:
        info = {"sample_rate": w.getframerate(), "channels": w.getnchannels(), "sample_width": w.getsampwidth(), "sample_count": w.getnframes()}
        data = w.readframes(info["sample_count"])
    return np.frombuffer(data, dtype="<i2").copy(), info


def wav_write(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(samples, dtype="<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(samples.tobytes())


def best_alignment(canonical, decoded):
    x = np.asarray(canonical, dtype=np.float64) - np.mean(canonical)
    y = np.asarray(decoded, dtype=np.float64) - np.mean(decoded)
    corr = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-len(x) + 1, len(y))
    mask = (lags >= -ALIGN_BOUND) & (lags <= ALIGN_BOUND)
    x2 = np.concatenate([[0.0], np.cumsum(x * x)])
    y2 = np.concatenate([[0.0], np.cumsum(y * y)])
    best = {"lag": None, "corr": -2.0, "post_count": 0, "leading_crop": "", "success": False}
    for lag, dot in zip(lags[mask], corr[mask]):
        lag = int(lag)
        xs = max(0, -lag)
        ys = max(0, lag)
        overlap = min(len(x) - xs, len(y) - ys)
        if overlap <= 256:
            continue
        denom = ((x2[xs + overlap] - x2[xs]) * (y2[ys + overlap] - y2[ys])) ** 0.5
        score = 0.0 if denom == 0 else float(dot / denom)
        if score > best["corr"]:
            post_possible = len(y) - ys
            success = post_possible >= len(x)
            best = {"lag": lag, "corr": score, "post_count": len(x) if success else max(0, post_possible), "leading_crop": ys, "success": bool(success)}
    return best


def c2_spec():
    edges = np.floor(np.linspace(0, 513, 65)).astype(np.int64)
    f_mod = np.fft.rfftfreq(372, d=256 / 48000)
    masks = {
        "MOD_BAND_1": np.where((f_mod > 0) & (f_mod <= 4))[0].tolist(),
        "MOD_BAND_2": np.where((f_mod > 4) & (f_mod <= 12))[0].tolist(),
        "MOD_BAND_3": np.where((f_mod > 12) & (f_mod <= 30))[0].tolist(),
        "MOD_BAND_4": np.where((f_mod > 30) & (f_mod <= np.max(f_mod)))[0].tolist(),
    }
    win = 0.5 - 0.5 * np.cos((2.0 * np.pi * np.arange(1024, dtype=np.float64)) / 1023.0)
    twin = 0.5 - 0.5 * np.cos((2.0 * np.pi * np.arange(372, dtype=np.float64)) / 371.0)
    return {
        "representation_id": REP_ID,
        "sample_rate": 48000,
        "sample_count": 96000,
        "float_conversion": "int16.astype(float64)/32768.0",
        "stft": {"n_fft": 1024, "win_length": 1024, "hop_length": 256, "center": False, "padding": "NONE", "frame_count": 372, "fft_bins": 513},
        "hann_convention": "symmetric Hann: 0.5 - 0.5*cos(2*pi*n/(N-1))",
        "power_normalization": "abs(fft)^2 / sum(window^2)",
        "epsilon": 1e-12,
        "frequency_group_edges": edges.tolist(),
        "frequency_group_count": 64,
        "temporal_de_meaning": "subtract mean over time per frequency group, no division by std",
        "temporal_hann_length": 372,
        "modulation_fft_length": 372,
        "modulation_frequency_grid": f_mod.tolist(),
        "modulation_nyquist_hz": 93.75,
        "modulation_band_masks": masks,
        "modulation_band_definitions": {"MOD_BAND_1": "0 < f <= 4 Hz", "MOD_BAND_2": "4 < f <= 12 Hz", "MOD_BAND_3": "12 < f <= 30 Hz", "MOD_BAND_4": "30 < f <= modulation Nyquist"},
        "flattening_order": "frequency group major, then modulation band 1..4",
        "expected_dimension": 256,
        "dtype": "float64",
        "window_sha256": hashlib.sha256(np.ascontiguousarray(win.astype("<f8")).tobytes()).hexdigest(),
        "temporal_window_sha256": hashlib.sha256(np.ascontiguousarray(twin.astype("<f8")).tobytes()).hexdigest(),
        "frequency_edges_sha256": hashlib.sha256(np.ascontiguousarray(edges.astype("<i8")).tobytes()).hexdigest(),
        "modulation_masks_sha256": sha256_text(json.dumps(masks, sort_keys=True)),
    }


EXTRACTOR_SOURCE = r'''import hashlib
import json
import wave
from pathlib import Path

import numpy as np


REPRESENTATION_ID = "LOGSPECTRAL_TEMPORAL_MODULATION_V1"


def read_pcm16_wav(path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        data = w.readframes(n)
    if sr != 48000 or ch != 1 or sw != 2 or n != 96000:
        raise ValueError(f"invalid PCM contract sr={sr} ch={ch} sw={sw} n={n}")
    return np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0


def symmetric_hann(n):
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * i) / float(n - 1))


def extract_from_pcm_float(x, spec):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (int(spec["sample_count"]),):
        raise ValueError(f"invalid input shape {x.shape}")
    n_fft = int(spec["stft"]["n_fft"])
    win_length = int(spec["stft"]["win_length"])
    hop = int(spec["stft"]["hop_length"])
    frames = int(spec["stft"]["frame_count"])
    edges = np.asarray(spec["frequency_group_edges"], dtype=np.int64)
    win = symmetric_hann(win_length)
    win_power = float(np.sum(win * win))
    grouped = np.empty((64, frames), dtype=np.float64)
    eps = float(spec["epsilon"])
    for t in range(frames):
        start = t * hop
        frame = x[start:start + win_length]
        spectrum = np.fft.rfft(frame * win, n=n_fft)
        power = (np.abs(spectrum) ** 2) / win_power
        logp = 10.0 * np.log10(np.maximum(power, eps))
        for b in range(64):
            grouped[b, t] = np.mean(logp[edges[b]:edges[b + 1]])
    residual = grouped - np.mean(grouped, axis=1, keepdims=True)
    twin = symmetric_hann(frames)
    tw_power = float(np.sum(twin * twin))
    f_mod = np.fft.rfftfreq(frames, d=hop / float(spec["sample_rate"]))
    masks = [
        (f_mod > 0) & (f_mod <= 4),
        (f_mod > 4) & (f_mod <= 12),
        (f_mod > 12) & (f_mod <= 30),
        (f_mod > 30) & (f_mod <= np.max(f_mod)),
    ]
    for m in masks:
        if not np.any(m):
            raise ValueError("empty modulation band")
    out = np.empty((64, 4), dtype=np.float64)
    for b in range(64):
        mc = np.fft.rfft(residual[b] * twin, n=frames)
        mp = (np.abs(mc) ** 2) / tw_power
        for q, mask in enumerate(masks):
            out[b, q] = 10.0 * np.log10(max(float(np.mean(mp[mask])), eps))
    feat = out.reshape(256).astype(np.float64, copy=False)
    if feat.shape != (256,) or not np.all(np.isfinite(feat)):
        raise ValueError("invalid feature")
    return feat


def extract_from_wav(path, spec):
    return extract_from_pcm_float(read_pcm16_wav(path), spec)


def canonical_array_hash(arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    payload = REPRESENTATION_ID.encode("ascii") + b"\n" + ",".join(str(x) for x in arr.shape).encode("ascii") + b"\n<f8\n" + arr.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()
'''


def load_c2_extractor(path):
    spec = importlib.util.spec_from_file_location("aq2c2_temporal_modulation_extractor", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def codec_info(codec, rate):
    if codec == "AAC":
        return {"encoder": "aac", "core": "FFmpeg-native AAC", "ext": "m4a", "probe": "aac", "args": ["-c:a", "aac", "-b:a", f"{rate}k"]}
    if codec == "MP3":
        return {"encoder": "libmp3lame", "core": "LAME", "ext": "mp3", "probe": "mp3", "args": ["-c:a", "libmp3lame", "-b:a", f"{rate}k"]}
    if codec == "Opus":
        return {"encoder": "libopus", "core": "libopus", "ext": "opus", "probe": "opus", "args": ["-c:a", "libopus", "-b:a", f"{rate}k", "-vbr", "off"]}
    raise ValueError(codec)


def metrics_for(rows):
    y_true = [r["true_codec"] for r in rows]
    y_pred = [r["predicted_codec"] for r in rows]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_codec_recall": {c: float(recall_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)) for c in CLASS_ORDER},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
        "n": len(rows),
        "class_order": CLASS_ORDER,
    }


def dominant_error(items, codec):
    counts = Counter(r["predicted_codec"] for r in items)
    errors = len(items) - counts.get(codec, 0)
    if errors == 0:
        return "NA", None
    wrong_counts = {c: counts.get(c, 0) for c in CLASS_ORDER if c != codec}
    target, count = max(wrong_counts.items(), key=lambda kv: kv[1])
    return target, float(count / errors)


def summary_stats(vals):
    vals = np.asarray(list(vals), dtype=np.float64)
    if vals.size == 0:
        return {"median": None, "iqr": None, "min": None, "max": None}
    return {"median": float(np.median(vals)), "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def extract_batch(manifest_path, spec_path, extractor_path, out_dir):
    manifest = pd.read_csv(manifest_path)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    mod = load_c2_extractor(Path(extractor_path))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    id_col = "derivative_uid" if "derivative_uid" in manifest.columns else "C2_parent_uid"
    path_col = "aligned_pcm_path" if "aligned_pcm_path" in manifest.columns else "canonical_pcm_path"
    for rec in manifest.to_dict("records"):
        uid = rec[id_col]
        arr = mod.extract_from_wav(rec[path_col], spec)
        out = out_dir / f"{uid}.npy"
        np.save(out, arr, allow_pickle=False)
        rows.append({"uid": uid, "array_sha256": mod.canonical_array_hash(arr), "npy_file_sha256": sha256_file(out), "shape": "256", "dtype": str(arr.dtype), "finite_check": bool(np.all(np.isfinite(arr))), "npy_path": str(out)})
    write_csv(out_dir / "manifest.csv", rows, list(rows[0].keys()))


def fail(subclassification, detail):
    write_json(ROOT / "03_manifests" / "AQ2C2_FREEZE.json", {
        "phase": "AQ-2C.2", "specification_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": "FAIL_INTEGRITY", "subclassification": subclassification, "detail": detail,
        "new_training_performed": "NO", "model_refit_or_retraining_of_B1_performed": "NO", "vorbis_used": "NO",
        "representation_A_primary_scientific_use": "NO", "representation_B_used": "NO", "remaining_blockers": []})
    raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-batch", action="store_true")
    ap.add_argument("--manifest")
    ap.add_argument("--spec")
    ap.add_argument("--extractor")
    ap.add_argument("--out-dir")
    args = ap.parse_args()
    if args.extract_batch:
        extract_batch(args.manifest, args.spec, args.extractor, args.out_dir)
        return

    for d in [
        "32_triangulation_corpus_c2/candidate_audit", "32_triangulation_corpus_c2/canonical_selected",
        "32_triangulation_corpus_c2/derivatives/encoded", "32_triangulation_corpus_c2/derivatives/decoded_raw",
        "32_triangulation_corpus_c2/derivatives/aligned_pcm", "32_triangulation_corpus_c2/clean",
        "32_triangulation_corpus_c2/scratch/source_extract", "32_triangulation_corpus_c2/scratch/candidate_canonical",
        "33_representation_c2/spec", "33_representation_c2/extractor", "33_representation_c2/training_features",
        "33_representation_c2/triangulation_features", "33_representation_c2/clean_features", "33_representation_c2/determinism",
        "34_probe_c2/model", "34_probe_c2/centroids", "34_probe_c2/freeze",
        "35_results_c2/predictions", "35_results_c2/per_rate", "35_results_c2/bootstrap", "35_results_c2/corpus",
        "35_results_c2/geometry", "35_results_c2/comparison", "35_results_c2/gate", "36_reports_c2", "logs_c2",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    files = {
        "AQ2C1B_FREEZE": ROOT / "03_manifests/AQ2C1B_FREEZE.json",
        "AQ2C1B_PREPREDICTION": ROOT / "03_manifests/AQ2C1B_PREPREDICTION_FREEZE.json",
        "C1B_SPEC": ROOT / "25_design_c1a/AQ2C1B_PROSPECTIVE_CONFIRMATION_SPEC.json",
        "DISCOVERY_EXCLUSION": ROOT / "23_confirmation_corpus_c1a/selection_audit/AQ2C1A_discovery_parent_exclusion.csv",
        "C1B_PARENT_MANIFEST": ROOT / "23_confirmation_corpus_c1a/AQ2C1A_confirmation_parent_manifest.csv",
        "C1A_CANDIDATE_INVENTORY": ROOT / "23_confirmation_corpus_c1a/candidate_inventory/AQ2C1A_candidate_source_inventory.csv",
        "C1B_SUMMARY": ROOT / "30_results_c1b/gate/AQ2C1B_confirmation_summary.json",
        "B1_DERIVATIVE_MANIFEST": ROOT / "16_derivatives_b1/AQ2B1_derivative_manifest.csv",
        "PYTHON": PYTHON,
    }
    for key, path in files.items():
        if sha256_file(path) != EXPECTED[key]:
            fail(f"{key}_IDENTITY_MISMATCH", {"actual_sha256": sha256_file(path)})

    c1b = json.loads(files["AQ2C1B_FREEZE"].read_text(encoding="utf-8"))
    if c1b.get("final_classification") != "PASS_NEW_RATE_BOUNDARY_CONFIRMED" or c1b.get("confirmation_parent_count") != 60 or c1b.get("actual_derivative_count") != 1080 or c1b.get("prediction_count") != 1080 or c1b.get("new_training_performed") != "NO" or c1b.get("model_refit_performed") != "NO" or c1b.get("new_model_weights_created") != "NO" or c1b.get("vorbis_used") != "NO" or c1b.get("representation_B_used") != "NO":
        fail("AQ2C1B_IDENTITY_MISMATCH", c1b)
    end = c1b["endpoint_metrics"]
    if abs(end["BA_ENDPOINT_POOLED"] - 0.8416666666666667) > 1e-15 or abs(end["AAC_RECALL_32"] - 0.9166666666666666) > 1e-15 or abs(end["AAC_RECALL_128"] - 0.9) > 1e-15 or end["ENDPOINT_CONTROL_PASS"] is not True or end["DOMAIN_SHIFT_SEVERE"] is not False:
        fail("AQ2C1B_IDENTITY_MISMATCH", {"endpoint": end})
    if c1b["full_new_rate_boundary_events"] != {"48": False, "80": True, "96": True} or c1b["replication_64"] is not True:
        fail("AQ2C1B_IDENTITY_MISMATCH", {"events": c1b["full_new_rate_boundary_events"], "rep64": c1b["replication_64"]})
    c1b_pre = json.loads(files["AQ2C1B_PREPREDICTION"].read_text(encoding="utf-8"))
    if c1b_pre.get("confirmation_prediction_count") != 0 or c1b_pre.get("scientific_confirmation_metrics_computed") != "NO" or c1b_pre.get("derivative_actual") != 1080 or c1b_pre.get("feature_success_count") != 1080 or c1b_pre.get("clean_feature_success_count") != 60:
        fail("AQ2C1B_PREPREDICTION_IDENTITY_MISMATCH", c1b_pre)

    r2 = json.loads((ROOT / "03_manifests/AQ2A_R2_FREEZE.json").read_text(encoding="utf-8"))
    encoder = Path(r2["r2_encoder_ffmpeg_path"])
    ffprobe = Path(r2["r2_ffprobe_path"])
    decoder = Path(r2["common_decoder_path"])
    if sha256_file(encoder) != EXPECTED["ENCODER"] or sha256_file(ffprobe) != EXPECTED["FFPROBE"] or sha256_file(decoder) != EXPECTED["DECODER"]:
        fail("TOOLCHAIN_IDENTITY_MISMATCH", {})
    if platform.python_version() != "3.12.3" or np.__version__ != "1.26.4" or scipy.__version__ != "1.13.1" or pd.__version__ != "2.2.2" or sklearn.__version__ != "1.5.1":
        fail("PYTHON_ENVIRONMENT_MISMATCH", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})

    discovery = pd.read_csv(files["DISCOVERY_EXCLUSION"])
    c1b_parents = pd.read_csv(files["C1B_PARENT_MANIFEST"])
    used_rows = []
    for rec in discovery.to_dict("records"):
        used_rows.append({"source_corpus": rec["source_corpus"], "parent_or_source_uid": rec["parent_uid"], "source_file_sha256": rec["source_file_sha256"], "canonical_pcm_sha256": rec["canonical_pcm_sha256"], "usage_role": "DISCOVERY"})
    for rec in c1b_parents.to_dict("records"):
        used_rows.append({"source_corpus": rec["source_corpus"], "parent_or_source_uid": rec["upstream_parent_uid"], "source_file_sha256": rec["source_file_sha256"], "canonical_pcm_sha256": rec["pcm_sha256"], "usage_role": "C1B_CONFIRMATION"})
    write_csv(ROOT / "32_triangulation_corpus_c2/candidate_audit/AQ2C2_used_parent_exclusion.csv", used_rows, list(used_rows[0].keys()))
    used_uid = set(r["parent_or_source_uid"] for r in used_rows)
    used_pcm = set(r["canonical_pcm_sha256"] for r in used_rows)
    if len(used_uid) != 142:
        fail("USED_PARENT_EXCLUSION_MISMATCH", {"unique_used_uid": len(used_uid)})

    candidates = pd.read_csv(files["C1A_CANDIDATE_INVENTORY"])
    candidates = candidates[(candidates["source_corpus"].isin(CORPORA)) & (candidates["locally_available"].astype(str) == "True") & (candidates["duration"].astype(float) >= 2.0)].copy()
    candidates["candidate_or_parent_uid"] = candidates["parent_uid"].astype(str)
    candidates = candidates[~candidates["candidate_or_parent_uid"].isin(used_uid)]
    candidates["C2_SELECTION_SCORE"] = candidates.apply(lambda r: sha256_text(f"AQ2C2_TRIANGULATION_SELECTION_V1|{r['source_corpus']}|{r['candidate_or_parent_uid']}|{r['source_file_sha256']}"), axis=1)
    selected = []
    selected_pcm = set()
    zip_cache = {}
    for corpus in CORPORA:
        rows = candidates[candidates["source_corpus"] == corpus].sort_values(["C2_SELECTION_SCORE", "candidate_or_parent_uid"]).to_dict("records")
        for rec in rows:
            if sum(1 for r in selected if r["source_corpus"] == corpus) >= 20:
                break
            zip_name, member = str(rec["absolute_source_path"]).split("!", 1)
            zip_path = Path(zip_name)
            src_tmp = ROOT / "32_triangulation_corpus_c2/scratch/source_extract" / f"{rec['existing_upstream_uid']}{Path(member).suffix or '.wav'}"
            cand_wav = ROOT / "32_triangulation_corpus_c2/scratch/candidate_canonical" / f"{rec['existing_upstream_uid']}.wav"
            if not cand_wav.exists():
                if zip_path not in zip_cache:
                    zip_cache[zip_path] = zipfile.ZipFile(zip_path)
                with zip_cache[zip_path].open(member) as src, open(src_tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                offset = max(0.0, (float(rec["duration"]) - 2.0) / 2.0)
                cmd = [str(decoder), "-y", "-hide_banner", "-ss", f"{offset:.9f}", "-i", str(src_tmp), "-map_metadata", "-1", "-vn", "-sn", "-dn", "-t", "2.0", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(cand_wav)]
                rr = run_cmd(cmd, timeout=120)
                try:
                    src_tmp.unlink()
                except FileNotFoundError:
                    pass
                if rr["return_code"] != 0:
                    continue
            arr, info = wav_read(cand_wav)
            pcm = raw_pcm_sha256(cand_wav)
            if info != {"sample_rate": 48000, "channels": 1, "sample_width": 2, "sample_count": 96000} or pcm in used_pcm or pcm in selected_pcm:
                continue
            selected_pcm.add(pcm)
            out = ROOT / "32_triangulation_corpus_c2/canonical_selected" / f"{corpus}_{rec['candidate_or_parent_uid'][:24]}.wav"
            shutil.copyfile(cand_wav, out)
            c2_uid = sha256_text(f"AQ2C2_PARENT_V1|{corpus}|{rec['candidate_or_parent_uid']}|{pcm}")
            selected.append({"C2_parent_uid": c2_uid, "source_corpus": corpus, "selection_score": rec["C2_SELECTION_SCORE"], "candidate_or_parent_uid": rec["candidate_or_parent_uid"], "source_file_sha256": rec["source_file_sha256"], "pcm_sha256": pcm, "sample_rate": 48000, "channels": 1, "sample_width_bits": 16, "sample_count": 96000, "duration_seconds": 2.0, "canonical_pcm_path": str(out), "absolute_source_path": rec["absolute_source_path"]})
    for z in zip_cache.values():
        z.close()
    if Counter(r["source_corpus"] for r in selected) != {"Orchset": 20, "IRMAS": 20, "FSDnoisy18k": 20}:
        fail("BLOCKED_TRIANGULATION_CORPUS", {"counts": Counter(r["source_corpus"] for r in selected)})
    parent_manifest_path = ROOT / "32_triangulation_corpus_c2/AQ2C2_parent_manifest.csv"
    write_csv(parent_manifest_path, sorted(selected, key=lambda r: (r["source_corpus"], r["selection_score"])), list(selected[0].keys()))
    with open(ROOT / "32_triangulation_corpus_c2/AQ2C2_parent_file_sha256s.txt", "w", encoding="utf-8", newline="\n") as f:
        for rec in sorted(selected, key=lambda r: r["canonical_pcm_path"]):
            p = Path(rec["canonical_pcm_path"])
            f.write(f"{sha256_file(p)}  {str(p.relative_to(ROOT)).replace('\\', '/')}  PCM_SHA256={rec['pcm_sha256']}\n")
    overlap_audit = {
        "C2_PARENT_COUNT": len(selected),
        "DISCOVERY_PARENT_OVERLAP": len(set(r["candidate_or_parent_uid"] for r in selected) & set(discovery["parent_or_source_uid"] if "parent_or_source_uid" in discovery.columns else discovery["parent_uid"])),
        "C1B_PARENT_OVERLAP": len(set(r["candidate_or_parent_uid"] for r in selected) & set(c1b_parents["upstream_parent_uid"])),
        "DISCOVERY_PCM_COLLISION": len(set(r["pcm_sha256"] for r in selected) & set(discovery["canonical_pcm_sha256"])),
        "C1B_PCM_COLLISION": len(set(r["pcm_sha256"] for r in selected) & set(c1b_parents["pcm_sha256"])),
        "WITHIN_C2_PCM_DUPLICATE": len(selected) - len(set(r["pcm_sha256"] for r in selected)),
        "parents_by_corpus": dict(Counter(r["source_corpus"] for r in selected)),
    }
    write_json(ROOT / "32_triangulation_corpus_c2/candidate_audit/AQ2C2_parent_overlap_audit.json", overlap_audit)
    if any(overlap_audit[k] for k in ["DISCOVERY_PARENT_OVERLAP", "C1B_PARENT_OVERLAP", "DISCOVERY_PCM_COLLISION", "C1B_PCM_COLLISION", "WITHIN_C2_PCM_DUPLICATE"]):
        fail("C2_PARENT_CONTAMINATION", overlap_audit)

    spec_obj = c2_spec()
    spec_path = ROOT / "33_representation_c2/spec/AQ2C2_representation_spec.json"
    extractor_path = ROOT / "33_representation_c2/extractor/aq2c2_temporal_modulation_extractor.py"
    write_json(spec_path, spec_obj)
    extractor_path.write_text(EXTRACTOR_SOURCE, encoding="utf-8")

    b1_deriv = pd.read_csv(files["B1_DERIVATIVE_MANIFEST"])
    train = b1_deriv[b1_deriv["split"] == "TRAIN"].copy()
    if len(train) != 294 or train["parent_uid"].nunique() != 49 or train["codec_family"].value_counts().to_dict() != {"AAC": 98, "MP3": 98, "Opus": 98}:
        fail("B1_TRAIN_INPUT_MISMATCH", {"count": len(train), "parents": train["parent_uid"].nunique(), "classes": train["codec_family"].value_counts().to_dict()})
    train_manifest_for_extract = ROOT / "33_representation_c2/determinism/AQ2C2_B1_train_input_manifest.csv"
    write_csv(train_manifest_for_extract, train[["derivative_uid", "aligned_pcm_path", "codec_family", "parent_uid"]].sort_values("derivative_uid").to_dict("records"), ["derivative_uid", "aligned_pcm_path", "codec_family", "parent_uid"])
    for i in [1, 2, 3]:
        out_dir = ROOT / f"33_representation_c2/determinism/RUN_{i}"
        cmd = [str(PYTHON), str(ROOT / "logs_c2/AQ2C2_execution_script.py"), "--extract-batch", "--manifest", str(train_manifest_for_extract), "--spec", str(spec_path), "--extractor", str(extractor_path), "--out-dir", str(out_dir)]
        rr = run_cmd(cmd, timeout=300)
        if rr["return_code"] != 0:
            fail("C2_REPRESENTATION_NONDETERMINISTIC", {"run": i, "stderr": rr["stderr"]})
    run1 = pd.read_csv(ROOT / "33_representation_c2/determinism/RUN_1/manifest.csv")
    det_rows = []
    for i in [2, 3]:
        other = pd.read_csv(ROOT / f"33_representation_c2/determinism/RUN_{i}/manifest.csv")
        m1 = dict(zip(run1["uid"], run1["npy_path"]))
        mo = dict(zip(other["uid"], other["npy_path"]))
        for uid in sorted(m1):
            a = np.load(m1[uid], allow_pickle=False)
            b = np.load(mo[uid], allow_pickle=False)
            det_rows.append({"uid": uid, "comparison": f"RUN_1_vs_RUN_{i}", "array_equal": bool(np.array_equal(a, b)), "max_abs_diff": float(np.max(np.abs(a - b))), "hash_equal": sha256_file(Path(m1[uid])) == sha256_file(Path(mo[uid]))})
    det_path = ROOT / "33_representation_c2/determinism/AQ2C2_representation_determinism.csv"
    write_csv(det_path, det_rows, list(det_rows[0].keys()))
    if len(det_rows) != 588 or not all(r["array_equal"] and r["max_abs_diff"] == 0 and r["hash_equal"] for r in det_rows):
        fail("C2_REPRESENTATION_NONDETERMINISTIC", {"rows": len(det_rows)})
    training_rows = []
    X_train = []
    y_train = []
    run1_by_uid = dict(zip(run1["uid"], run1["npy_path"]))
    for rec in train.sort_values("derivative_uid").to_dict("records"):
        arr = np.load(run1_by_uid[rec["derivative_uid"]], allow_pickle=False)
        out = ROOT / "33_representation_c2/training_features" / f"{rec['derivative_uid']}.npy"
        np.save(out, arr, allow_pickle=False)
        training_rows.append({"derivative_uid": rec["derivative_uid"], "codec_family": rec["codec_family"], "parent_uid": rec["parent_uid"], "npy_path": str(out), "array_sha256": load_c2_extractor(extractor_path).canonical_array_hash(arr), "npy_file_sha256": sha256_file(out), "shape": "256", "dtype": str(arr.dtype), "finite_check": True})
        X_train.append(arr)
        y_train.append(rec["codec_family"])
    X_train = np.vstack(X_train)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z_train = scaler.fit_transform(X_train)
    clf = RidgeClassifier(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=1e-4, class_weight=None, solver="svd", positive=False, random_state=None)
    clf.fit(Z_train, y_train)
    if list(clf.classes_) != CLASS_ORDER:
        fail("C2_CLASS_ORDER_MISMATCH", {"classes": list(clf.classes_)})
    probe_numeric_path = ROOT / "34_probe_c2/model/AQ2C2_probe_numeric.npz"
    np.savez(probe_numeric_path, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_, ridge_coef=clf.coef_, ridge_intercept=clf.intercept_, class_order=np.asarray(CLASS_ORDER))
    probe_config = {"scaler": {"type": "StandardScaler", "with_mean": True, "with_std": True, "fit_scope": "B1_TRAIN_REPRESENTATION_C_ONLY"}, "classifier": {"type": "RidgeClassifier", "alpha": 1.0, "fit_intercept": True, "copy_X": True, "max_iter": None, "tol": 1e-4, "class_weight": None, "solver": "svd", "positive": False, "random_state": None}, "class_order": CLASS_ORDER, "training_rows": 294, "training_parents": 49, "packages": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    probe_config_path = ROOT / "34_probe_c2/model/AQ2C2_probe_config.json"
    write_json(probe_config_path, probe_config)
    centroids = {c: np.mean(Z_train[np.asarray(y_train) == c], axis=0) for c in CLASS_ORDER}
    centroid_path = ROOT / "34_probe_c2/centroids/AQ2C2_train_centroids.npz"
    np.savez(centroid_path, centroid_AAC=centroids["AAC"], centroid_MP3=centroids["MP3"], centroid_Opus=centroids["Opus"], class_order=np.asarray(CLASS_ORDER))
    probe_freeze_path = ROOT / "03_manifests/AQ2C2_PROBE_FREEZE.json"
    write_json(probe_freeze_path, {"phase": "AQ-2C.2", "created_utc": datetime.now(timezone.utc).isoformat(), "AQ2C1B_freeze_sha256": EXPECTED["AQ2C1B_FREEZE"], "C2_parent_manifest_sha256": sha256_file(parent_manifest_path), "representation_spec_sha256": sha256_file(spec_path), "representation_extractor_sha256": sha256_file(extractor_path), "B1_train_derivative_manifest_sha256": EXPECTED["B1_DERIVATIVE_MANIFEST"], "B1_train_parent_count": 49, "B1_train_row_count": 294, "representation_determinism_status": "PASS", "scaler_configuration": probe_config["scaler"], "RidgeClassifier_configuration": probe_config["classifier"], "probe_numeric_sha256": sha256_file(probe_numeric_path), "probe_config_sha256": sha256_file(probe_config_path), "centroid_numeric_sha256": sha256_file(centroid_path), "C2_parent_count": 60, "C2_prediction_count": 0, "C2_scientific_metrics_computed": "NO", "C1B_data_used_for_training": "NO", "C2_data_used_for_training": "NO"})
    probe_freeze_sha = sha256_file(probe_freeze_path)

    source_samples = {}
    derivative_rows = []
    parents = pd.read_csv(parent_manifest_path)
    for rec in parents.sort_values(["source_corpus", "C2_parent_uid"]).to_dict("records"):
        parent = rec["C2_parent_uid"]
        src = Path(rec["canonical_pcm_path"])
        source_samples[parent] = wav_read(src)[0]
        for codec in CLASS_ORDER:
            for rate in RATES:
                ci = codec_info(codec, rate)
                uid_source = f"AQ2C2_DERIV_V1|{parent}|{codec}|{rate}|{ci['core']}|{EXPECTED['ENCODER']}"
                duid = sha256_text(uid_source)[:24]
                enc = ROOT / "32_triangulation_corpus_c2/derivatives/encoded" / f"{duid}.{ci['ext']}"
                dec = ROOT / "32_triangulation_corpus_c2/derivatives/decoded_raw" / f"{duid}.wav"
                ali = ROOT / "32_triangulation_corpus_c2/derivatives/aligned_pcm" / f"{duid}.wav"
                enc_r = run_cmd([str(encoder), "-y", "-hide_banner", "-i", str(src), "-map_metadata", "-1", "-vn", "-sn", "-dn"] + ci["args"] + [str(enc)], timeout=120)
                enc_ok = enc_r["return_code"] == 0 and enc.exists()
                probe_cmd = [str(ffprobe), "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels,duration,bit_rate:format=format_name,duration,bit_rate", "-of", "json", str(enc)]
                probe_r = run_cmd(probe_cmd, timeout=60) if enc_ok else {"command": " ".join(probe_cmd), "return_code": 1, "stdout": "", "stderr": "encode failed"}
                try:
                    js = json.loads(probe_r["stdout"]) if probe_r["return_code"] == 0 else {}
                except json.JSONDecodeError:
                    js = {}
                stream = js.get("streams", [{}])[0] if js.get("streams") else {}
                probe_ok = probe_r["return_code"] == 0 and stream.get("codec_name", "") == ci["probe"]
                dec_cmd = [str(decoder), "-y", "-hide_banner", "-i", str(enc), "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(dec)]
                dec_r = run_cmd(dec_cmd, timeout=120) if probe_ok else {"command": " ".join(dec_cmd), "return_code": 1, "stdout": "", "stderr": "probe failed"}
                dec_ok = dec_r["return_code"] == 0 and dec.exists()
                align = {"lag": "", "corr": "", "post_count": "", "leading_crop": "", "success": False}
                aligned_sha = ""
                if dec_ok:
                    dec_samp, dec_info = wav_read(dec)
                    if dec_info["sample_rate"] == 48000 and dec_info["channels"] == 1 and dec_info["sample_width"] == 2:
                        align = best_alignment(source_samples[parent], dec_samp)
                        if align["success"] and align["post_count"] == SEGMENT_SAMPLES:
                            start = int(align["leading_crop"])
                            wav_write(ali, dec_samp[start:start + SEGMENT_SAMPLES])
                            aligned_sha = raw_pcm_sha256(ali)
                ok = enc_ok and probe_ok and dec_ok and ali.exists() and align["success"] and align["post_count"] == SEGMENT_SAMPLES
                derivative_rows.append({"derivative_uid": duid, "C2_parent_uid": parent, "source_corpus": rec["source_corpus"], "codec_family": codec, "encoder_name": ci["encoder"], "encoder_core": ci["core"], "nominal_rate_kbps": rate, "achieved_rate_kbps": (8 * enc.stat().st_size / SEGMENT_SECONDS / 1000.0) if enc.exists() else "", "encoded_sha256": sha256_file(enc) if enc.exists() else "", "aligned_pcm_sha256": aligned_sha, "alignment_lag_samples": align["lag"], "aligned_sample_count": SEGMENT_SAMPLES if ok else "", "encoded_size_bytes": enc.stat().st_size if enc.exists() else "", "encoded_path": str(enc) if enc.exists() else "", "aligned_pcm_path": str(ali) if ali.exists() else "", "derivative_id_source_string": uid_source, "encode_return_code": enc_r["return_code"], "probe_return_code": probe_r["return_code"], "decode_return_code": dec_r["return_code"], "probe_codec_name": stream.get("codec_name", ""), "status": "PASS" if ok else "FAIL", "encode_command": enc_r["command"], "probe_command": probe_r["command"], "decode_command": dec_r["command"]})
    deriv_path = ROOT / "32_triangulation_corpus_c2/derivatives/AQ2C2_derivative_manifest.csv"
    write_csv(deriv_path, derivative_rows, list(derivative_rows[0].keys()))
    if len(derivative_rows) != 1080 or not all(r["status"] == "PASS" for r in derivative_rows) or len({r["derivative_uid"] for r in derivative_rows}) != 1080:
        fail("C2_DERIVATIVE_MATRIX_INCOMPLETE", {"rows": len(derivative_rows), "pass": sum(r["status"] == "PASS" for r in derivative_rows)})
    with open(ROOT / "32_triangulation_corpus_c2/derivatives/AQ2C2_derivative_file_sha256s.txt", "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(list((ROOT / "32_triangulation_corpus_c2/derivatives/encoded").glob("*")) + list((ROOT / "32_triangulation_corpus_c2/derivatives/decoded_raw").glob("*")) + list((ROOT / "32_triangulation_corpus_c2/derivatives/aligned_pcm").glob("*")), key=lambda x: str(x.relative_to(ROOT)).replace("\\", "/")):
            f.write(f"{sha256_file(p)}  {str(p.relative_to(ROOT)).replace('\\', '/')}\n")
    coll = []
    by_pcm = defaultdict(list)
    for r in derivative_rows:
        by_pcm[r["aligned_pcm_sha256"]].append(r)
    for pcm, rows in by_pcm.items():
        if len(rows) > 1:
            coll.append({"aligned_pcm_sha256": pcm, "count": len(rows), "codec_labels": ";".join(sorted({r["codec_family"] for r in rows})), "rates": ";".join(map(str, sorted({int(r["nominal_rate_kbps"]) for r in rows}))), "derivative_uids": ";".join(r["derivative_uid"] for r in rows)})
    if not coll:
        coll = [{"aligned_pcm_sha256": "", "count": 0, "codec_labels": "", "rates": "", "derivative_uids": ""}]
    coll_path = ROOT / "32_triangulation_corpus_c2/derivatives/AQ2C2_collision_audit.csv"
    write_csv(coll_path, coll, list(coll[0].keys()))
    ach_rows = []
    for codec in CLASS_ORDER:
        for rate in RATES:
            vals = [float(r["achieved_rate_kbps"]) for r in derivative_rows if r["codec_family"] == codec and int(r["nominal_rate_kbps"]) == rate]
            ach_rows.append({"codec": codec, "nominal_rate_kbps": rate, **summary_stats(vals), "n": len(vals)})
    ach_path = ROOT / "32_triangulation_corpus_c2/derivatives/AQ2C2_achieved_rate_summary.csv"
    write_csv(ach_path, ach_rows, list(ach_rows[0].keys()))

    mod = load_c2_extractor(extractor_path)
    spec_loaded = json.loads(spec_path.read_text(encoding="utf-8"))
    feature_rows = []
    feature_by_uid = {}
    for rec in derivative_rows:
        arr = mod.extract_from_wav(rec["aligned_pcm_path"], spec_loaded)
        out = ROOT / "33_representation_c2/triangulation_features" / f"{rec['derivative_uid']}.npy"
        np.save(out, arr, allow_pickle=False)
        feature_by_uid[rec["derivative_uid"]] = arr
        feature_rows.append({"derivative_uid": rec["derivative_uid"], "array_sha256": mod.canonical_array_hash(arr), "npy_file_sha256": sha256_file(out), "shape": "256", "dtype": str(arr.dtype), "finite_check": True, "npy_path": str(out)})
    feat_path = ROOT / "33_representation_c2/AQ2C2_feature_manifest.csv"
    write_csv(feat_path, feature_rows, list(feature_rows[0].keys()))
    clean_rows = []
    clean_by_parent = {}
    for rec in parents.to_dict("records"):
        arr = mod.extract_from_wav(rec["canonical_pcm_path"], spec_loaded)
        out = ROOT / "33_representation_c2/clean_features" / f"{rec['C2_parent_uid']}.npy"
        np.save(out, arr, allow_pickle=False)
        clean_by_parent[rec["C2_parent_uid"]] = arr
        clean_rows.append({"C2_parent_uid": rec["C2_parent_uid"], "source_corpus": rec["source_corpus"], "array_sha256": mod.canonical_array_hash(arr), "npy_file_sha256": sha256_file(out), "shape": "256", "dtype": str(arr.dtype), "finite_check": True, "npy_path": str(out)})
    clean_feat_path = ROOT / "33_representation_c2/AQ2C2_clean_feature_manifest.csv"
    write_csv(clean_feat_path, clean_rows, list(clean_rows[0].keys()))
    with open(ROOT / "33_representation_c2/AQ2C2_feature_file_sha256s.txt", "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(list((ROOT / "33_representation_c2/training_features").glob("*.npy")) + list((ROOT / "33_representation_c2/triangulation_features").glob("*.npy")) + list((ROOT / "33_representation_c2/clean_features").glob("*.npy")), key=lambda x: str(x.relative_to(ROOT)).replace("\\", "/")):
            f.write(f"{sha256_file(p)}  {str(p.relative_to(ROOT)).replace('\\', '/')}\n")

    prepred_path = ROOT / "03_manifests/AQ2C2_PREPREDICTION_FREEZE.json"
    write_json(prepred_path, {"phase": "AQ-2C.2", "created_utc": datetime.now(timezone.utc).isoformat(), "AQ2C1B_freeze_sha256": EXPECTED["AQ2C1B_FREEZE"], "AQ2C2_probe_freeze_sha256": probe_freeze_sha, "C2_parent_count": 60, "parents_per_corpus": 20, "C2_overlap_counts": overlap_audit, "expected_derivatives": 1080, "actual_derivatives": 1080, "encode_success": 1080, "probe_success": 1080, "decode_success": 1080, "alignment_success": 1080, "feature_success": len(feature_rows), "clean_feature_success": len(clean_rows), "representation_identity": REP_ID, "probe_identity": "StandardScaler + RidgeClassifier(alpha=1.0, solver=svd)", "centroid_identity": sha256_file(centroid_path), "C2_prediction_count": 0, "scientific_metrics_computed": "NO"})
    prepred_sha = sha256_file(prepred_path)

    model = np.load(probe_numeric_path, allow_pickle=False)
    m_mean, m_scale, ridge_coef, ridge_intercept = model["scaler_mean"], model["scaler_scale"], model["ridge_coef"], model["ridge_intercept"]
    centroid_np = np.load(centroid_path, allow_pickle=False)
    cents = {"AAC": centroid_np["centroid_AAC"], "MP3": centroid_np["centroid_MP3"], "Opus": centroid_np["centroid_Opus"]}
    pred_rows = []
    z_by_uid = {}
    for rec in derivative_rows:
        x = feature_by_uid[rec["derivative_uid"]]
        z = (x - m_mean) / m_scale
        z_by_uid[rec["derivative_uid"]] = z
        scores = (z[None, :] @ ridge_coef.T + ridge_intercept)[0]
        pred = CLASS_ORDER[int(np.argmax(scores))]
        true = rec["codec_family"]
        ti = CLASS_ORDER.index(true)
        dists = {c: float(np.linalg.norm(z - cents[c])) for c in CLASS_ORDER}
        dtrue = dists[true]
        rank = 1 + sum(1 for c in CLASS_ORDER if c != true and dists[c] < dtrue)
        wrong_closer = min(dists[c] for c in CLASS_ORDER if c != true) < dtrue
        pred_rows.append({"derivative_uid": rec["derivative_uid"], "C2_parent_uid": rec["C2_parent_uid"], "source_corpus": rec["source_corpus"], "true_codec": true, "nominal_rate_kbps": int(rec["nominal_rate_kbps"]), "predicted_codec": pred, "score_AAC": float(scores[0]), "score_MP3": float(scores[1]), "score_Opus": float(scores[2]), "true_margin": float(scores[ti] - max(scores[j] for j in range(3) if j != ti)), "distance_centroid_AAC": dists["AAC"], "distance_centroid_MP3": dists["MP3"], "distance_centroid_Opus": dists["Opus"], "true_centroid_rank": rank, "wrong_centroid_closer": bool(wrong_closer), "achieved_rate_kbps": float(rec["achieved_rate_kbps"])})
    pred_path = ROOT / "35_results_c2/predictions/AQ2C2_all_predictions.csv"
    write_csv(pred_path, pred_rows, list(pred_rows[0].keys()))
    per_rate = {str(rate): metrics_for([r for r in pred_rows if r["nominal_rate_kbps"] == rate]) for rate in RATES}
    per_rate_path = ROOT / "35_results_c2/per_rate/AQ2C2_per_rate_metrics.json"
    write_json(per_rate_path, per_rate)
    codec_rate_rows = []
    for codec in CLASS_ORDER:
        for rate in RATES:
            items = [r for r in pred_rows if r["true_codec"] == codec and r["nominal_rate_kbps"] == rate]
            target, share = dominant_error(items, codec)
            codec_rate_rows.append({"codec": codec, "nominal_rate_kbps": rate, "n": len(items), "recall": float(np.mean([r["predicted_codec"] == codec for r in items])), "median_TRUE_MARGIN": float(np.median([r["true_margin"] for r in items])), "fraction_TRUE_MARGIN_gt_0": float(np.mean([r["true_margin"] > 0 for r in items])), "fraction_TRUE_CENTROID_RANK_1": float(np.mean([r["true_centroid_rank"] == 1 for r in items])), "fraction_WRONG_CENTROID_CLOSER": float(np.mean([r["wrong_centroid_closer"] for r in items])), "dominant_error_target": target, "dominant_error_share": share})
    codec_rate_path = ROOT / "35_results_c2/per_rate/AQ2C2_per_codec_rate_summary.csv"
    write_csv(codec_rate_path, codec_rate_rows, list(codec_rate_rows[0].keys()))
    endpoint_metrics = metrics_for([r for r in pred_rows if r["nominal_rate_kbps"] in [32, 128]])
    aac32 = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == 32)["recall"]
    aac128 = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == 128)["recall"]
    floor = min(aac32, aac128)
    endpoint_pass = endpoint_metrics["balanced_accuracy"] >= 0.75 and aac32 >= 0.60 and aac128 >= 0.60
    severe = endpoint_metrics["balanced_accuracy"] < 0.65 or (aac32 < 0.50 and aac128 < 0.50)
    parent_aac = {}
    for p in parents["C2_parent_uid"]:
        parent_aac[p] = {rate: next(r for r in pred_rows if r["C2_parent_uid"] == p and r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == rate)["predicted_codec"] == "AAC" for rate in RATES}
    boot_summary = {}
    for rate in MID_RATES:
        rng = np.random.default_rng(20261001 + rate)
        pids = list(parent_aac)
        drops = np.asarray([0.5 * (parent_aac[p][32] + parent_aac[p][128]) - parent_aac[p][rate] for p in pids], dtype=np.float64)
        reps = []
        for i in range(10000):
            idx = rng.integers(0, len(pids), len(pids))
            reps.append({"replicate": i, "mean_drop": float(np.mean(drops[idx]))})
        bp = ROOT / f"35_results_c2/bootstrap/AQ2C2_AAC_bootstrap_{rate}.csv"
        write_csv(bp, reps, ["replicate", "mean_drop"])
        vals = np.asarray([r["mean_drop"] for r in reps])
        boot_summary[str(rate)] = {"seed": 20261001 + rate, "p2_5": float(np.percentile(vals, 2.5)), "p50": float(np.percentile(vals, 50)), "p97_5": float(np.percentile(vals, 97.5)), "C2_PAIRED_DROP_STABLE": bool(np.percentile(vals, 2.5) > 0), "replicates": 10000}
    boot_summary_path = ROOT / "35_results_c2/bootstrap/AQ2C2_AAC_bootstrap_summary.json"
    write_json(boot_summary_path, boot_summary)
    corpus_rows = []
    for corpus in CORPORA:
        for rate in RATES:
            items = [r for r in pred_rows if r["source_corpus"] == corpus and r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == rate]
            corpus_rows.append({"source_corpus": corpus, "nominal_rate_kbps": rate, "AAC_recall": float(np.mean([r["predicted_codec"] == "AAC" for r in items])), "n": len(items)})
    corpus_path = ROOT / "35_results_c2/corpus/AQ2C2_AAC_corpus_rate_summary.csv"
    write_csv(corpus_path, corpus_rows, list(corpus_rows[0].keys()))
    c_lookup = {(r["source_corpus"], r["nominal_rate_kbps"]): r["AAC_recall"] for r in corpus_rows}
    multi = {}
    for rate in MID_RATES:
        details, ct = {}, 0
        for corpus in CORPORA:
            deg = c_lookup[(corpus, rate)] < 0.5 * (c_lookup[(corpus, 32)] + c_lookup[(corpus, 128)])
            details[corpus] = bool(deg)
            ct += int(deg)
        multi[str(rate)] = {"corpus_degradation": details, "MULTI_CORPUS_TRUE_COUNT": ct, "C2_MULTI_CORPUS_CONFIRMATION": bool(ct >= 2)}
    multi_path = ROOT / "35_results_c2/corpus/AQ2C2_multi_corpus_gate.json"
    write_json(multi_path, multi)
    events = {}
    for rate in MID_RATES:
        cr = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == rate)
        drop = floor - cr["recall"]
        structural = cr["recall"] <= 0.50 and drop >= 0.25 and cr["median_TRUE_MARGIN"] < 0 and cr["fraction_WRONG_CENTROID_CLOSER"] >= 0.60 and cr["dominant_error_share"] is not None and cr["dominant_error_share"] >= 0.60
        events[str(rate)] = {"AAC_RECALL": cr["recall"], "AAC_RECALL_DROP": drop, "AAC_MEDIAN_TRUE_MARGIN": cr["median_TRUE_MARGIN"], "AAC_WRONG_CENTROID_FRACTION": cr["fraction_WRONG_CENTROID_CLOSER"], "AAC_DOMINANT_ERROR_TARGET": cr["dominant_error_target"], "AAC_DOMINANT_ERROR_SHARE": cr["dominant_error_share"], "C2_STRUCTURAL_EVENT": bool(structural), "BOOTSTRAP_LOW": boot_summary[str(rate)]["p2_5"], "BOOTSTRAP_MEDIAN": boot_summary[str(rate)]["p50"], "BOOTSTRAP_HIGH": boot_summary[str(rate)]["p97_5"], "C2_PAIRED_DROP_STABLE": boot_summary[str(rate)]["C2_PAIRED_DROP_STABLE"], "MULTI_CORPUS_TRUE_COUNT": multi[str(rate)]["MULTI_CORPUS_TRUE_COUNT"], "C2_MULTI_CORPUS_CONFIRMATION": multi[str(rate)]["C2_MULTI_CORPUS_CONFIRMATION"], "FULL_C2_TRIANGULATION_EVENT": bool(structural and boot_summary[str(rate)]["C2_PAIRED_DROP_STABLE"] and multi[str(rate)]["C2_MULTI_CORPUS_CONFIRMATION"])}
    full_rates = [int(r) for r, e in events.items() if e["FULL_C2_TRIANGULATION_EVENT"]]
    if severe or not endpoint_pass:
        final, subclass = "TRIANGULATION_PROBE_INADEQUATE", ""
    elif len(full_rates) >= 2 and any(r in [80, 96] for r in full_rates):
        final, subclass = "PASS_BOUNDARY_TRIANGULATED", ""
    elif len(full_rates) == 1 or sum(e["C2_STRUCTURAL_EVENT"] for e in events.values()) >= 2:
        final, subclass = "PARTIAL_BOUNDARY_TRIANGULATION", ""
    else:
        final, subclass = "BOUNDARY_NOT_TRIANGULATED", ""
    if final == "PASS_BOUNDARY_TRIANGULATED":
        interpretation = "The configuration-conditioned AAC family-attribution instability persisted under a separately frozen temporal-modulation representation and independent Ridge probe on a second untouched parent cohort, providing cross-representation and cross-probe empirical triangulation."
    elif final == "PARTIAL_BOUNDARY_TRIANGULATION":
        interpretation = "Some aspects of the configuration-conditioned instability transferred to the independent representation/probe, but the full triangulation criterion was not met."
    elif final == "BOUNDARY_NOT_TRIANGULATED":
        interpretation = "The C1B configuration-conditioned instability was not reproduced under the prospectively frozen second representation/probe despite adequate endpoint competence."
    else:
        interpretation = "The new representation/probe does not establish sufficient endpoint attribution competence for a meaningful configuration-boundary triangulation."
    clean_rows_out = []
    for rec in derivative_rows:
        x = feature_by_uid[rec["derivative_uid"]]
        xc = clean_by_parent[rec["C2_parent_uid"]]
        z = (x - m_mean) / m_scale
        zc = (xc - m_mean) / m_scale
        clean_rows_out.append({"derivative_uid": rec["derivative_uid"], "C2_parent_uid": rec["C2_parent_uid"], "source_corpus": rec["source_corpus"], "codec": rec["codec_family"], "nominal_rate_kbps": int(rec["nominal_rate_kbps"]), "D_CLEAN_RAW": float(np.linalg.norm(x - xc)), "D_CLEAN_STD": float(np.linalg.norm(z - zc))})
    clean_dist_path = ROOT / "35_results_c2/geometry/AQ2C2_clean_distance.csv"
    write_csv(clean_dist_path, clean_rows_out, list(clean_rows_out[0].keys()))
    uid_lookup = {(r["C2_parent_uid"], r["codec_family"], int(r["nominal_rate_kbps"])): r["derivative_uid"] for r in derivative_rows}
    traj_rows = []
    for p in parents["C2_parent_uid"]:
        corpus = str(parents.loc[parents["C2_parent_uid"] == p, "source_corpus"].iloc[0])
        for codec in CLASS_ORDER:
            z32 = z_by_uid[uid_lookup[(p, codec, 32)]]
            z128 = z_by_uid[uid_lookup[(p, codec, 128)]]
            v = z128 - z32
            denom = float(np.dot(v, v))
            endpoint = float(np.linalg.norm(v))
            for rate in MID_RATES:
                z = z_by_uid[uid_lookup[(p, codec, rate)]]
                if denom == 0:
                    alpha = resid = norm = within = None
                else:
                    alpha = float(np.dot(z - z32, v) / denom)
                    proj = z32 + alpha * v
                    resid = float(np.linalg.norm(z - proj))
                    norm = resid / endpoint
                    within = bool(0 <= alpha <= 1)
                traj_rows.append({"C2_parent_uid": p, "source_corpus": corpus, "codec": codec, "nominal_rate_kbps": rate, "ALPHA": alpha, "ORTHOGONAL_RESIDUAL": resid, "ENDPOINT_DISTANCE": endpoint, "NORMALIZED_RESIDUAL": norm, "WITHIN_ENDPOINT_SEGMENT": within})
    traj_path = ROOT / "35_results_c2/geometry/AQ2C2_endpoint_trajectory_geometry.csv"
    write_csv(traj_path, traj_rows, list(traj_rows[0].keys()))
    c1b_summary = json.loads(files["C1B_SUMMARY"].read_text(encoding="utf-8"))
    c1b_aac = [c1b_summary["per_rate"][str(r)]["AAC_recall"] for r in RATES]
    c2_aac = [per_rate[str(r)]["per_codec_recall"]["AAC"] for r in RATES]
    rho, pval = spearmanr(c1b_aac, c2_aac)
    comparison = {"rates": RATES, "C1B_AAC_recall": c1b_aac, "C2_AAC_recall": c2_aac, "C1B_per_rate_BA": [c1b_summary["per_rate"][str(r)]["balanced_accuracy"] for r in RATES], "C2_per_rate_BA": [per_rate[str(r)]["balanced_accuracy"] for r in RATES], "C2_full_event_status": {r: events[str(r)]["FULL_C2_TRIANGULATION_EVENT"] for r in MID_RATES}, "C1B_C2_AAC_recall_spearman_rho": float(rho), "C1B_C2_AAC_recall_spearman_p_value": float(pval)}
    comp_path = ROOT / "35_results_c2/comparison/AQ2C2_C1B_profile_comparison.json"
    write_json(comp_path, comparison)
    gate = {"ENDPOINT": {"BA_ENDPOINT_POOLED_C2": endpoint_metrics["balanced_accuracy"], "AAC_RECALL_32_C2": aac32, "AAC_RECALL_128_C2": aac128, "AAC_ENDPOINT_RECALL_FLOOR_C2": floor, "C2_ENDPOINT_CONTROL_PASS": bool(endpoint_pass), "C2_ENDPOINT_SEVERE_FAIL": bool(severe)}, **{str(r): events[str(r)] for r in MID_RATES}, "full_event_count": len(full_rates), "full_event_rates": full_rates, "full_event_contains_80_or_96": any(r in [80, 96] for r in full_rates), "final_classification": final, "maximum_permitted_interpretation": interpretation}
    gate_path = ROOT / "35_results_c2/gate/AQ2C2_triangulation_gate.json"
    write_json(gate_path, gate)
    aac_profile = {str(rate): next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == rate) for rate in RATES}
    triang_summary = {"C2_parent_count": 60, "parents_by_corpus": dict(Counter(parents["source_corpus"])), "rate_grid": RATES, "codec_set": CLASS_ORDER, "per_rate": per_rate, "AAC_rate_profile": aac_profile, "bootstrap_summary": boot_summary, "multi_corpus_summary": multi, "full_C2_triangulation_events": {str(r): events[str(r)]["FULL_C2_TRIANGULATION_EVENT"] for r in MID_RATES}, "C1B_profile_comparison": comparison, "final_classification": final}
    summary_path = ROOT / "35_results_c2/gate/AQ2C2_triangulation_summary.json"
    write_json(summary_path, triang_summary)

    artifact_paths = [
        parent_manifest_path, ROOT / "32_triangulation_corpus_c2/AQ2C2_parent_file_sha256s.txt",
        ROOT / "32_triangulation_corpus_c2/candidate_audit/AQ2C2_used_parent_exclusion.csv",
        ROOT / "32_triangulation_corpus_c2/candidate_audit/AQ2C2_parent_overlap_audit.json",
        deriv_path, ROOT / "32_triangulation_corpus_c2/derivatives/AQ2C2_derivative_file_sha256s.txt", coll_path, ach_path,
        spec_path, extractor_path, det_path, ROOT / "33_representation_c2/AQ2C2_feature_manifest.csv", clean_feat_path,
        ROOT / "33_representation_c2/AQ2C2_feature_file_sha256s.txt", probe_numeric_path, probe_config_path, centroid_path,
        probe_freeze_path, prepred_path, pred_path, per_rate_path, codec_rate_path, boot_summary_path, corpus_path, multi_path,
        clean_dist_path, traj_path, comp_path, gate_path, summary_path, ROOT / "logs_c2/AQ2C2_execution_script.py",
    ]
    for rate in MID_RATES:
        artifact_paths.append(ROOT / f"35_results_c2/bootstrap/AQ2C2_AAC_bootstrap_{rate}.csv")
    artifact_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in artifact_paths if p.exists()}
    freeze = {
        "phase": "AQ-2C.2", "specification_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": final, "subclassification": subclass, "AQ2C1B_freeze_sha256": EXPECTED["AQ2C1B_FREEZE"],
        "AQ2C1B_preprediction_freeze_sha256": EXPECTED["AQ2C1B_PREPREDICTION"], "AQ2C2_probe_freeze_sha256": probe_freeze_sha,
        "AQ2C2_preprediction_freeze_sha256": prepred_sha, "C2_parent_manifest_sha256": sha256_file(parent_manifest_path),
        "C2_parent_count": 60, "parents_per_corpus": 20, "discovery_parent_overlap": overlap_audit["DISCOVERY_PARENT_OVERLAP"],
        "C1B_parent_overlap": overlap_audit["C1B_PARENT_OVERLAP"], "discovery_pcm_collision": overlap_audit["DISCOVERY_PCM_COLLISION"],
        "C1B_pcm_collision": overlap_audit["C1B_PCM_COLLISION"], "within_C2_pcm_duplicate": overlap_audit["WITHIN_C2_PCM_DUPLICATE"],
        "representation_id": REP_ID, "representation_spec_sha256": sha256_file(spec_path), "representation_extractor_sha256": sha256_file(extractor_path),
        "representation_determinism_status": "PASS", "probe_type": "StandardScaler + RidgeClassifier", "probe_config": probe_config,
        "probe_numeric_sha256": sha256_file(probe_numeric_path), "centroid_numeric_sha256": sha256_file(centroid_path),
        "B1_train_parent_count": 49, "B1_train_row_count": 294, "C1B_data_used_for_training": "NO", "C2_data_used_for_training": "NO",
        "rate_grid": RATES, "codec_set": CLASS_ORDER, "expected_derivative_count": 1080, "actual_derivative_count": 1080,
        "encode_success_count": 1080, "probe_success_count": 1080, "decode_success_count": 1080, "alignment_success_count": 1080,
        "feature_success_count": len(feature_rows), "clean_feature_success_count": len(clean_rows), "endpoint_metrics": gate["ENDPOINT"],
        "per_rate_metrics": per_rate, "AAC_rate_profile": aac_profile, "bootstrap_summary": boot_summary, "multi_corpus_summary": multi,
        "full_C2_triangulation_events": {str(r): events[str(r)]["FULL_C2_TRIANGULATION_EVENT"] for r in MID_RATES},
        "C1B_profile_comparison": comparison, "new_training_performed": "YES", "model_refit_or_retraining_of_B1_performed": "NO",
        "new_C2_probe_training_performed": "YES", "vorbis_used": "NO", "representation_A_primary_scientific_use": "NO",
        "representation_B_used": "NO", "AQ2A_v1_frozen_artifacts_modified": "NO", "AQ2A_R1_frozen_artifacts_modified": "NO",
        "AQ2A_R2_frozen_artifacts_modified": "NO", "AQ2B0_frozen_artifacts_modified": "NO", "AQ2B1_frozen_artifacts_modified": "NO",
        "AQ2B1_DX_frozen_artifacts_modified": "NO", "AQ2C1A_frozen_artifacts_modified": "NO", "AQ2C1B_frozen_artifacts_modified": "NO",
        "existing_MM_artifacts_modified": "NO", "remaining_blockers": [], "artifact_hashes": artifact_hashes,
    }
    freeze_path = ROOT / "03_manifests/AQ2C2_FREEZE.json"
    write_json(freeze_path, freeze)
    artifact_hashes[str(freeze_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(freeze_path)
    report_path = ROOT / "36_reports_c2/AQ2C2_EXECUTION_REPORT.md"
    sections = [
        ("1. Final C2 classification", final), ("2. Subclassification", subclass), ("3. Maximum permitted scientific interpretation", interpretation),
        ("4. AQ-2C.1B identity reconciliation", "PASS"), ("5. C1B prospective result preservation", "AQ-2C.1B remains PASS_NEW_RATE_BOUNDARY_CONFIRMED."),
        ("6. Complete 142-parent exclusion construction", "PASS"), ("7. C2 candidate-pool identity", "PASS; C1A candidate inventory used."),
        ("8. Deterministic C2 selection method", "SHA256 AQ2C2_TRIANGULATION_SELECTION_V1 sorting, then eligibility gates."),
        ("9. Selected parent count", 60), ("10. Parent counts by corpus", json.dumps(dict(Counter(parents["source_corpus"])), indent=2)),
        ("11. Discovery-parent overlap", overlap_audit["DISCOVERY_PARENT_OVERLAP"]), ("12. C1B-parent overlap", overlap_audit["C1B_PARENT_OVERLAP"]),
        ("13. Discovery PCM collision", overlap_audit["DISCOVERY_PCM_COLLISION"]), ("14. C1B PCM collision", overlap_audit["C1B_PCM_COLLISION"]),
        ("15. Within-C2 PCM duplicate count", overlap_audit["WITHIN_C2_PCM_DUPLICATE"]), ("16. New representation motivation", "Temporal modulation structure rather than stationary spectral mean/std statistics."),
        ("17. Exact Representation-C specification", json.dumps(spec_obj, indent=2)), ("18. Representation-C spec SHA-256", sha256_file(spec_path)),
        ("19. Extractor SHA-256", sha256_file(extractor_path)), ("20. Three-run determinism result", "PASS; 588 exact checks."),
        ("21. B1 TRAIN row count", 294), ("22. B1 TRAIN class counts", json.dumps(dict(Counter(y_train)), indent=2)),
        ("23. Explicit confirmation C1B was not used for C2 training", "C1B DATA USED FOR C2 TRAINING = NO"),
        ("24. C2 StandardScaler contract", json.dumps(probe_config["scaler"], indent=2)), ("25. C2 RidgeClassifier contract", json.dumps(probe_config["classifier"], indent=2)),
        ("26. C2 probe numeric SHA-256", sha256_file(probe_numeric_path)), ("27. C2 centroid SHA-256", sha256_file(centroid_path)),
        ("28. C2 PROBE FREEZE SHA-256", probe_freeze_sha), ("29. Explicit C2 prediction count at probe freeze = 0", "0"),
        ("30. C2 derivative expected/actual", "1080 / 1080"), ("31. Encode/probe/decode/alignment integrity", "1080 / 1080 / 1080 / 1080"),
        ("32. Representation-C feature integrity", f"{len(feature_rows)} / 1080"), ("33. Clean-feature integrity", f"{len(clean_rows)} / 60"),
        ("34. C2 PREPREDICTION FREEZE SHA-256", prepred_sha), ("35. Explicit C2 prediction count at preprediction freeze = 0", "0"),
        ("36. Per-rate balanced accuracy", json.dumps({r: per_rate[str(r)]["balanced_accuracy"] for r in RATES}, indent=2)),
        ("37. Per-rate macro-F1", json.dumps({r: per_rate[str(r)]["macro_f1"] for r in RATES}, indent=2)),
        ("38. Per-codec recall", json.dumps({r: per_rate[str(r)]["per_codec_recall"] for r in RATES}, indent=2)),
        ("39. C2 endpoint pooled BA", endpoint_metrics["balanced_accuracy"]), ("40. AAC recall 32", aac32), ("41. AAC recall 128", aac128),
        ("42. Endpoint-control classification", json.dumps(gate["ENDPOINT"], indent=2)),
        ("43. AAC 48 structural values", json.dumps(events["48"], indent=2)), ("44. AAC 64 structural values", json.dumps(events["64"], indent=2)),
        ("45. AAC 80 structural values", json.dumps(events["80"], indent=2)), ("46. AAC 96 structural values", json.dumps(events["96"], indent=2)),
        ("47. Parent-bootstrap 48", json.dumps(boot_summary["48"], indent=2)), ("48. Parent-bootstrap 64", json.dumps(boot_summary["64"], indent=2)),
        ("49. Parent-bootstrap 80", json.dumps(boot_summary["80"], indent=2)), ("50. Parent-bootstrap 96", json.dumps(boot_summary["96"], indent=2)),
        ("51. Multi-corpus 48", json.dumps(multi["48"], indent=2)), ("52. Multi-corpus 64", json.dumps(multi["64"], indent=2)),
        ("53. Multi-corpus 80", json.dumps(multi["80"], indent=2)), ("54. Multi-corpus 96", json.dumps(multi["96"], indent=2)),
        ("55. Full C2 event status by rate", json.dumps({r: events[str(r)]["FULL_C2_TRIANGULATION_EVENT"] for r in MID_RATES}, indent=2)),
        ("56. Full event count", len(full_rates)), ("57. Whether at least one full event is at 80/96", any(r in [80, 96] for r in full_rates)),
        ("58. Clean-distance diagnostic", "See 35_results_c2/geometry/AQ2C2_clean_distance.csv."),
        ("59. Endpoint-trajectory geometry", "See 35_results_c2/geometry/AQ2C2_endpoint_trajectory_geometry.csv."),
        ("60. C1B-versus-C2 AAC profile comparison", json.dumps(comparison, indent=2)), ("61. Final triangulation-gate calculation", json.dumps(gate, indent=2)),
        ("62. Scientific interpretation", interpretation), ("63. Whether phenomenon is now cross-representation/probe triangulated", "YES" if final == "PASS_BOUNDARY_TRIANGULATED" else "NO"),
        ("64. Whether any model-independent claim is authorized", "NO"), ("65. All artifacts", "\n".join(sorted(artifact_hashes))),
        ("66. Major SHA-256 values", json.dumps({**EXPECTED, **artifact_hashes}, indent=2)), ("67. Remaining blockers", "None"),
        ("68. Explicit statement", "AQ-2B.1 RESULT REMAINS =\nSTOP_MFEA1_TRACE_NOT_SUPPORTED"), ("69. Explicit statement", "AQ-2C.1B RESULT REMAINS =\nPASS_NEW_RATE_BOUNDARY_CONFIRMED"),
        ("70. Explicit statement", "C2 TRIANGULATION PARENT COUNT = 60"), ("71. Explicit statement", "DISCOVERY PARENT OVERLAP = 0"),
        ("72. Explicit statement", "C1B CONFIRMATION PARENT OVERLAP = 0"), ("73. Explicit statement", "C1B DATA USED FOR C2 TRAINING = NO"),
        ("74. Explicit statement", "C2 DATA USED FOR C2 TRAINING = NO"), ("75. Explicit statement", "NEW C2 PROBE TRAINING PERFORMED = YES"),
        ("76. Explicit statement", "B1 MODEL REFIT/RETRAINING PERFORMED = NO"), ("77. Explicit statement", "REPRESENTATION A USED AS PRIMARY C2 REPRESENTATION = NO"),
        ("78. Explicit statement", "REPRESENTATION B USED = NO"), ("79. Explicit statement", "VORBIS USED = NO"),
        ("80. Explicit statements", "AQ2A V1 FROZEN ARTIFACTS MODIFIED = NO\nAQ2A R1 FROZEN ARTIFACTS MODIFIED = NO\nAQ2A R2 FROZEN ARTIFACTS MODIFIED = NO\nAQ2B0 FROZEN ARTIFACTS MODIFIED = NO\nAQ2B1 FROZEN ARTIFACTS MODIFIED = NO\nAQ2B1-DX FROZEN ARTIFACTS MODIFIED = NO\nAQ2C1A FROZEN ARTIFACTS MODIFIED = NO\nAQ2C1B FROZEN ARTIFACTS MODIFIED = NO\nEXISTING MM ARTIFACTS MODIFIED = NO"),
    ]
    lines = ["# AQ-2C.2 Execution Report", ""]
    for title, body in sections:
        lines.extend([f"## {title}", str(body), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifact_hashes[str(report_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(report_path)
    cmd_log = ROOT / "logs_c2/AQ2C2_command_log.txt"
    cmd_log.write_text(json.dumps({"phase": "AQ-2C.2", "created_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "cwd": str(ROOT), "environment": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}, "final_classification": final}, indent=2) + "\n", encoding="utf-8")
    artifact_hashes[str(cmd_log.relative_to(ROOT)).replace("\\", "/")] = sha256_file(cmd_log)
    with open(ROOT / "AQ2C2_SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as f:
        for rel, h in sorted(artifact_hashes.items()):
            f.write(f"{h}  {rel}\n")
    print(json.dumps({"final_classification": final, "subclassification": subclass, "endpoint_BA": endpoint_metrics["balanced_accuracy"], "AAC_RECALL_32": aac32, "AAC_RECALL_128": aac128, "full_event_rates": full_rates, "prediction_count": len(pred_rows), "ledger_sha256": sha256_file(ROOT / "AQ2C2_SHA256SUMS.txt")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
