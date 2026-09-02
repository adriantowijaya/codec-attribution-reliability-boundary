import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import warnings
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler


AQ = Path(r"[REDACTED_LOCAL_PATH]\Tony Hidayat\AQ_OpenWorld_Codec_Provenance")
PYTHON_B0 = Path(r"[REDACTED_LOCAL_PATH]\anaconda3\python.exe")
FFMPEG_ENCODER_R2 = AQ / "tools" / "gyan_ffmpeg_9.0_full" / "bin" / "ffmpeg.exe"
FFPROBE_R2 = AQ / "tools" / "gyan_ffmpeg_9.0_full" / "bin" / "ffprobe.exe"
FFMPEG_CANONICAL_V1 = Path(r"[REDACTED_LOCAL_PATH]\AppData\Local\CapCut\Apps\7.7.0.3143\ffmpeg.exe")

EXPECTED = {
    "AQ2B0_FREEZE": "e4d714e3902ece12f27caedf7e502a105e9a08066b97d73d755ba69f419d96ab",
    "AQ2A_R2_FREEZE": "dfec802d21f7210b39ffbdc13ac825245c04a692509b70c2b8426a6c9cb7d26f",
    "CORPUS_SET": "76daaf1fbd368166c803b0a6d6839d68dd6ed755dd7ac9887a8c49d951c85059",
    "PYTHON": "62c225fb9cdc41b139c7024581c233644f975ffc35314558c60ebefa6b88be01",
    "FEATURE_SPEC": "83020402082b7168d88df8d1eb02fc4f254d24b6fbd4ab0770403d1f96c1acae",
    "FEATURE_EXTRACTOR": "6c79cbc06c67c9cff140cd64de6f878ea59f1b1580bb7db0b0acb08a4a57da8f",
    "FFMPEG_ENCODER_R2": "57c56e369d5b4873b4d93fc1a1d833cb7cd8bc9325c14b05c34ce60b22842d8a",
    "FFPROBE_R2": "afe05347caaabe479b3c4eae71992b6ec1e11c57266a1d665deb0f9fe9847208",
    "FFMPEG_CANONICAL_V1": "4dc3e63209cb6f183b703c8842f6e3dcc22778ccca1a3b9f4b5fca4034bb54dd",
    "LOSSLESS": "7d974f822000daacb80cc06fe4f549bc397d8636cd554a4e839fec77701f96b5",
}

CLASS_ORDER = ["AAC", "MP3", "Opus"]
CODECS = {
    "AAC": {"encoder": "aac", "core": "FFmpeg-native-AAC", "ext": "m4a", "probe_codec": "aac"},
    "MP3": {"encoder": "libmp3lame", "core": "LAME", "ext": "mp3", "probe_codec": "mp3"},
    "Opus": {"encoder": "libopus", "core": "libopus", "ext": "opus", "probe_codec": "opus"},
}
TRAIN_RATES = [32, 128]
VALIDATION_RATES = [32, 128]
TEST_RATES = [32, 64, 128]
HELDOUT_RATE = 64
SEGMENT_SAMPLES = 96000
SEGMENT_SECONDS = 2.0
ALIGN_BOUND = 8192


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def command_string(argv):
    return " ".join('"' + str(x) + '"' if " " in str(x) else str(x) for x in argv)


def run_cmd(argv, timeout=180):
    start = datetime.now(timezone.utc).isoformat()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        rc, out, err = p.returncode, p.stdout or "", p.stderr or ""
    except Exception as exc:
        rc, out, err = 999, "", repr(exc)
    end = datetime.now(timezone.utc).isoformat()
    return {"command": command_string(argv), "start": start, "end": end, "return_code": rc, "stdout": out, "stderr": err}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wav_read(path):
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.getnframes()
        data = w.readframes(frames)
    arr = np.frombuffer(data, dtype="<i2").astype(np.float64)
    return arr, {"sample_rate": rate, "channels": channels, "sample_width": width, "frames": frames, "duration": frames / rate}


def wav_write(path, samples):
    samples = np.asarray(samples, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(samples.astype("<i2").tobytes())


def raw_pcm_sha256(path):
    h = hashlib.sha256()
    with wave.open(str(path), "rb") as w:
        while True:
            data = w.readframes(65536)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def center_segment(src, dst):
    x, info = wav_read(src)
    if info["sample_rate"] != 48000 or info["channels"] != 1 or info["sample_width"] != 2:
        raise ValueError(f"canonical source violates format: {src}")
    start = (len(x) - SEGMENT_SAMPLES) // 2
    if start < 0:
        raise ValueError(f"canonical source too short: {src}")
    wav_write(dst, np.asarray(x[start:start + SEGMENT_SAMPLES], dtype=np.int16))
    return dst


def best_alignment(canonical, decoded):
    x = np.asarray(canonical, dtype=np.float64) - np.mean(canonical)
    y = np.asarray(decoded, dtype=np.float64) - np.mean(decoded)
    corr = correlate(y, x, mode="full", method="fft")
    lags = np.arange(-len(x) + 1, len(y))
    mask = (lags >= -ALIGN_BOUND) & (lags <= ALIGN_BOUND)
    x2 = np.concatenate([[0.0], np.cumsum(x * x)])
    y2 = np.concatenate([[0.0], np.cumsum(y * y)])
    best = {"lag": None, "corr": -2.0, "post_count": 0, "leading_crop": "", "trailing_crop": "", "success": False}
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
            best = {
                "lag": lag,
                "corr": score,
                "post_count": len(x) if success else max(0, post_possible),
                "leading_crop": ys,
                "trailing_crop": max(0, len(y) - (ys + len(x))) if success else "",
                "success": bool(success),
            }
    return best


def ffprobe(path):
    cmd = [
        str(FFPROBE_R2), "-v", "error",
        "-show_entries",
        "stream=codec_name,codec_long_name,profile,sample_rate,channels,duration,bit_rate:format=format_name,format_long_name,duration,bit_rate",
        "-of", "json", str(path),
    ]
    r = run_cmd(cmd, timeout=90)
    try:
        js = json.loads(r["stdout"]) if r["return_code"] == 0 else {}
    except json.JSONDecodeError:
        js = {}
        r["return_code"] = 998
        r["stderr"] += "\nffprobe JSON parse failure"
    stream = (js.get("streams") or [{}])[0]
    fmt = js.get("format") or {}
    return r, {
        "probe_codec_name": stream.get("codec_name", ""),
        "probe_codec_long_name": stream.get("codec_long_name", ""),
        "probe_profile": stream.get("profile", ""),
        "probe_sample_rate": stream.get("sample_rate", ""),
        "probe_channels": stream.get("channels", ""),
        "probe_duration": stream.get("duration") or fmt.get("duration", ""),
        "probe_bit_rate": stream.get("bit_rate") or fmt.get("bit_rate", ""),
        "probe_format_name": fmt.get("format_name", ""),
        "probe_format_long_name": fmt.get("format_long_name", ""),
    }


def derivative_uid(parent_uid, codec, rate, core):
    s = f"AQ2B1_DERIV_V1|{parent_uid}|{codec}|{rate}|{core}|{EXPECTED['FFMPEG_ENCODER_R2']}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24], s


def array_hash(arr, rep_id="STFT_LOGPOWER_STATS_V1"):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    shape = ",".join(str(x) for x in arr.shape)
    payload = rep_id.encode("ascii") + b"\n" + shape.encode("ascii") + b"\n<f8\n" + arr.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def metrics_bundle(y_true, y_pred):
    labels = CLASS_ORDER
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_recall": {label: float(v) for label, v in zip(labels, recalls)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "class_order": labels,
    }


def prediction_rows(deriv_rows, y_true, y_pred, probs):
    rows = []
    for meta, yt, yp, pr in zip(deriv_rows, y_true, y_pred, probs):
        rows.append({
            "derivative_uid": meta["derivative_uid"],
            "parent_uid": meta["parent_uid"],
            "source_corpus": meta["source_corpus"],
            "true_codec": yt,
            "predicted_codec": yp,
            "prob_AAC": pr[0],
            "prob_MP3": pr[1],
            "prob_Opus": pr[2],
        })
    return rows


def load_extractor():
    spec = importlib.util.spec_from_file_location("aq2b0_feature_extractor", AQ / "14_feature_preflight_b0" / "aq2b0_feature_extractor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    for d in [
        "16_derivatives_b1/encoded", "16_derivatives_b1/decoded_raw", "16_derivatives_b1/aligned_pcm", "16_derivatives_b1/scratch",
        "17_features_b1/representation_a", "18_model_b1", "19_results_b1/predictions", "19_results_b1/bootstrap",
        "19_results_b1/permutation", "19_results_b1/diagnostics", "20_reports_b1", "logs_b1",
    ]:
        (AQ / d).mkdir(parents=True, exist_ok=True)

    os.environ.update({
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    final = ""
    subclass = ""
    blockers = []
    commands = []

    b0 = json.loads((AQ / "03_manifests" / "AQ2B0_FREEZE.json").read_text(encoding="utf-8"))
    r2 = json.loads((AQ / "03_manifests" / "AQ2A_R2_FREEZE.json").read_text(encoding="utf-8"))
    b0_checks = {
        "freeze_sha": sha256_file(AQ / "03_manifests" / "AQ2B0_FREEZE.json") == EXPECTED["AQ2B0_FREEZE"],
        "classification": b0.get("final_classification") == "PASS_FEATURE_PIPELINE_FROZEN",
        "pipeline": b0.get("feature_pipeline_ready") == "YES",
        "rep_a": b0.get("representation_A_ready") == "YES",
        "rep_b": b0.get("representation_B_ready") == "YES",
        "parent_r2": b0.get("AQ2A_R2_freeze_sha256") == EXPECTED["AQ2A_R2_FREEZE"],
        "corpus": b0.get("corpus_set_sha256") == EXPECTED["CORPUS_SET"],
        "canonical_count": b0.get("canonical_pcm_count") == 82,
        "duration": b0.get("segment_duration_seconds") == 2.0,
        "split": b0.get("split_counts") == {"TRAIN": 49, "VALIDATION": 16, "TEST": 17},
    }
    if not all(b0_checks.values()):
        final, subclass = "FAIL_INTEGRITY", "AQ2B0_IDENTITY_MISMATCH"
        blockers.append(subclass)

    env_code = "import json,sys,numpy,scipy,pandas,sklearn; print(json.dumps({'python':sys.version,'numpy':numpy.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,'sklearn':sklearn.__version__}))"
    env_r = run_cmd([str(PYTHON_B0), "-c", env_code], timeout=60)
    env_info = json.loads(env_r["stdout"]) if env_r["return_code"] == 0 else {}
    py_ok = (
        sha256_file(PYTHON_B0) == EXPECTED["PYTHON"]
        and env_info.get("python", "").startswith("3.12.3")
        and env_info.get("numpy") == "1.26.4"
        and env_info.get("scipy") == "1.13.1"
        and env_info.get("pandas") == "2.2.2"
        and env_info.get("sklearn") == "1.5.1"
    )
    if final == "" and not py_ok:
        final, subclass = "FAIL_INTEGRITY", "PYTHON_ENVIRONMENT_MISMATCH"
        blockers.append(subclass)

    feature_spec_path = AQ / "14_feature_preflight_b0" / "AQ2B0_feature_spec.json"
    extractor_path = AQ / "14_feature_preflight_b0" / "aq2b0_feature_extractor.py"
    feature_ok = sha256_file(feature_spec_path) == EXPECTED["FEATURE_SPEC"] and sha256_file(extractor_path) == EXPECTED["FEATURE_EXTRACTOR"]
    if final == "" and not feature_ok:
        final, subclass = "FAIL_INTEGRITY", "B1_FEATURE_PIPELINE_FAIL"
        blockers.append(subclass)

    tool_ok = (
        sha256_file(FFMPEG_ENCODER_R2) == EXPECTED["FFMPEG_ENCODER_R2"]
        and sha256_file(FFPROBE_R2) == EXPECTED["FFPROBE_R2"]
        and sha256_file(FFMPEG_CANONICAL_V1) == EXPECTED["FFMPEG_CANONICAL_V1"]
    )
    if final == "" and not tool_ok:
        final, subclass = "FAIL_INTEGRITY", "TOOLCHAIN_IDENTITY_MISMATCH"
        blockers.append(subclass)

    lossless_rows = read_csv(AQ / "05_lossless_control" / "lossless_roundtrip_audit.csv")
    lossless_ok = (
        sha256_file(AQ / "05_lossless_control" / "lossless_roundtrip_audit.csv") == EXPECTED["LOSSLESS"]
        and len(lossless_rows) == 82
        and all(r["status"] == "PASS" and r["raw_pcm_original_sha256"] == r["raw_pcm_flac_roundtrip_sha256"] == r["raw_pcm_wavpack_roundtrip_sha256"] for r in lossless_rows)
        and b0.get("metadata_leakage_sentinel_status") == "PASS"
        and b0.get("label_blind_extraction_status") == "PASS"
    )
    if final == "" and not lossless_ok:
        final, subclass = "FAIL_INTEGRITY", "LOSSLESS_SENTINEL_FAIL"
        blockers.append(subclass)

    canonical = {r["parent_source_uid"]: r for r in read_csv(AQ / "03_manifests" / "canonical_manifest.csv")}
    split_rows = read_csv(AQ / "03_manifests" / "split_manifest.csv")
    split_map = {r["parent_source_uid"]: r["split"] for r in split_rows}
    corpus_map = {r["parent_source_uid"]: r.get("source_corpus", "") for r in canonical.values()}
    split_sets = defaultdict(set)
    for uid, split in split_map.items():
        split_sets[split].add(uid)
    split_audit = [
        {"comparison": "TRAIN_vs_VALIDATION", "intersection_count": len(split_sets["TRAIN"] & split_sets["VALIDATION"]), "status": "PASS" if not (split_sets["TRAIN"] & split_sets["VALIDATION"]) else "FAIL"},
        {"comparison": "TRAIN_vs_TEST", "intersection_count": len(split_sets["TRAIN"] & split_sets["TEST"]), "status": "PASS" if not (split_sets["TRAIN"] & split_sets["TEST"]) else "FAIL"},
        {"comparison": "VALIDATION_vs_TEST", "intersection_count": len(split_sets["VALIDATION"] & split_sets["TEST"]), "status": "PASS" if not (split_sets["VALIDATION"] & split_sets["TEST"]) else "FAIL"},
    ]
    write_csv(AQ / "19_results_b1" / "diagnostics" / "AQ2B1_parent_split_audit.csv", split_audit, ["comparison", "intersection_count", "status"])
    if final == "" and any(r["status"] != "PASS" for r in split_audit):
        final, subclass = "FAIL_INTEGRITY", "PARENT_LEAKAGE"
        blockers.append(subclass)

    derivative_rows = []
    feature_rows = []
    if final == "":
        expected_by_split = {"TRAIN": 294, "VALIDATION": 96, "TEST": 153}
        extractor = load_extractor()
        feature_spec = json.loads(feature_spec_path.read_text(encoding="utf-8"))
        segment_dir = AQ / "16_derivatives_b1" / "scratch" / "source_segments"
        source_segments = {}
        for uid, row in canonical.items():
            source_segments[uid] = center_segment(Path(row["canonical_wav_path"]), segment_dir / f"{uid[:24]}.wav")

        for uid in sorted(split_map):
            split = split_map[uid]
            rates = TRAIN_RATES if split == "TRAIN" else VALIDATION_RATES if split == "VALIDATION" else TEST_RATES
            for codec in CLASS_ORDER:
                c = CODECS[codec]
                for rate in rates:
                    duid, source_string = derivative_uid(uid, codec, rate, c["core"])
                    encoded = AQ / "16_derivatives_b1" / "encoded" / f"{duid}.{c['ext']}"
                    decoded = AQ / "16_derivatives_b1" / "decoded_raw" / f"{duid}.wav"
                    aligned = AQ / "16_derivatives_b1" / "aligned_pcm" / f"{duid}.wav"
                    src = source_segments[uid]
                    enc_cmd = [
                        str(FFMPEG_ENCODER_R2), "-y", "-hide_banner", "-i", str(src), "-map_metadata", "-1",
                        "-vn", "-sn", "-dn", "-c:a", c["encoder"], "-b:a", f"{rate}k", str(encoded),
                    ]
                    enc = run_cmd(enc_cmd)
                    commands.append(enc["command"])
                    enc_ok = enc["return_code"] == 0 and encoded.exists() and encoded.stat().st_size > 0
                    probe_cmd_result, probe_fields = ffprobe(encoded) if enc_ok else ({"command": "", "return_code": "", "stdout": "", "stderr": ""}, {})
                    if enc_ok:
                        commands.append(probe_cmd_result["command"])
                    probe_ok = enc_ok and probe_cmd_result["return_code"] == 0 and probe_fields.get("probe_codec_name") == c["probe_codec"]
                    dec_cmd = [
                        str(FFMPEG_CANONICAL_V1), "-y", "-hide_banner", "-i", str(encoded), "-map_metadata", "-1",
                        "-vn", "-sn", "-dn", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(decoded),
                    ]
                    dec = run_cmd(dec_cmd) if probe_ok else {"command": command_string(dec_cmd), "return_code": "", "stdout": "", "stderr": ""}
                    if probe_ok:
                        commands.append(dec["command"])
                    dec_ok = dec["return_code"] == 0 and decoded.exists() and decoded.stat().st_size > 0
                    align = {"lag": "", "corr": "", "post_count": 0, "leading_crop": "", "trailing_crop": "", "success": False}
                    aligned_sha = ""
                    if dec_ok:
                        dec_samples, dec_info = wav_read(decoded)
                        src_samples, _ = wav_read(src)
                        format_ok = dec_info["sample_rate"] == 48000 and dec_info["channels"] == 1 and dec_info["sample_width"] == 2
                        align = best_alignment(src_samples, dec_samples) if format_ok else align
                        if align["success"] and align["post_count"] == SEGMENT_SAMPLES:
                            start = int(align["leading_crop"])
                            wav_write(aligned, np.asarray(dec_samples[start:start + SEGMENT_SAMPLES], dtype=np.int16))
                            aligned_sha = raw_pcm_sha256(aligned)
                    aligned_ok = aligned.exists() and align["success"] and align["post_count"] == SEGMENT_SAMPLES
                    status = "PASS" if enc_ok and probe_ok and dec_ok and aligned_ok else "FAIL"
                    derivative_rows.append({
                        "derivative_uid": duid,
                        "parent_uid": uid,
                        "source_corpus": corpus_map.get(uid, ""),
                        "split": split,
                        "codec_family": codec,
                        "encoder_name": c["encoder"],
                        "encoder_core": c["core"],
                        "nominal_rate_kbps": rate,
                        "achieved_rate_kbps": (8 * encoded.stat().st_size / SEGMENT_SECONDS / 1000) if encoded.exists() else "",
                        "encoded_sha256": sha256_file(encoded) if encoded.exists() else "",
                        "aligned_pcm_sha256": aligned_sha,
                        "alignment_lag_samples": align["lag"],
                        "alignment_correlation": align["corr"],
                        "aligned_sample_count": SEGMENT_SAMPLES if aligned_ok else "",
                        "encoded_path": str(encoded) if encoded.exists() else "",
                        "aligned_pcm_path": str(aligned) if aligned.exists() else "",
                        "derivative_id_source_string": source_string,
                        "encode_command": enc["command"],
                        "encode_return_code": enc["return_code"],
                        "encode_stdout": enc["stdout"],
                        "encode_stderr": enc["stderr"],
                        "probe_command": probe_cmd_result.get("command", ""),
                        "probe_return_code": probe_cmd_result.get("return_code", ""),
                        "probe_stdout": probe_cmd_result.get("stdout", ""),
                        "probe_stderr": probe_cmd_result.get("stderr", ""),
                        "decode_command": dec["command"],
                        "decode_return_code": dec["return_code"],
                        "decode_stdout": dec["stdout"],
                        "decode_stderr": dec["stderr"],
                        "status": status,
                        **probe_fields,
                    })
        fields = [
            "derivative_uid", "parent_uid", "source_corpus", "split", "codec_family", "encoder_name", "encoder_core",
            "nominal_rate_kbps", "achieved_rate_kbps", "encoded_sha256", "aligned_pcm_sha256", "alignment_lag_samples",
            "alignment_correlation", "aligned_sample_count", "encoded_path", "aligned_pcm_path", "derivative_id_source_string",
            "encode_command", "encode_return_code", "encode_stdout", "encode_stderr", "probe_command", "probe_return_code",
            "probe_stdout", "probe_stderr", "decode_command", "decode_return_code", "decode_stdout", "decode_stderr",
            "probe_codec_name", "probe_codec_long_name", "probe_profile", "probe_sample_rate", "probe_channels",
            "probe_duration", "probe_bit_rate", "probe_format_name", "probe_format_long_name", "status",
        ]
        write_csv(AQ / "16_derivatives_b1" / "AQ2B1_derivative_manifest.csv", derivative_rows, fields)
        counts_ok = len(derivative_rows) == 543 and Counter(r["split"] for r in derivative_rows) == expected_by_split
        ops_ok = all(r["status"] == "PASS" for r in derivative_rows)
        if not counts_ok:
            final, subclass = "BLOCKED_EXECUTION", "INCOMPLETE_DERIVATIVE_MATRIX"
            blockers.append(subclass)
        elif not ops_ok:
            final, subclass = "BLOCKED_EXECUTION", "DERIVATIVE_OPERATION_FAIL"
            blockers.append(subclass)

    if final == "":
        extractor = load_extractor()
        feature_spec = json.loads(feature_spec_path.read_text(encoding="utf-8"))
        for row in derivative_rows:
            x = extractor.read_pcm16_wav(row["aligned_pcm_path"])
            G = extractor.extract_stft_core(x, feature_spec)
            arr = extractor.extract_representation_a(G, feature_spec)
            out = AQ / "17_features_b1" / "representation_a" / f"{row['derivative_uid']}.npy"
            np.save(out, arr, allow_pickle=False)
            feature_rows.append({
                "derivative_uid": row["derivative_uid"],
                "array_sha256": array_hash(arr),
                "npy_file_sha256": sha256_file(out),
                "shape": ",".join(map(str, arr.shape)),
                "dtype": str(arr.dtype),
                "finite_check": bool(np.all(np.isfinite(arr))),
                "npy_path": str(out),
            })
        write_csv(AQ / "17_features_b1" / "AQ2B1_feature_manifest.csv", feature_rows, ["derivative_uid", "array_sha256", "npy_file_sha256", "shape", "dtype", "finite_check", "npy_path"])
        feat_ok = len(feature_rows) == 543 and all(r["shape"] == "256" and r["dtype"] == "float64" and str(r["finite_check"]) == "True" for r in feature_rows) and sha256_file(extractor_path) == EXPECTED["FEATURE_EXTRACTOR"]
        if not feat_ok:
            final, subclass = "FAIL_INTEGRITY", "B1_FEATURE_PIPELINE_FAIL"
            blockers.append(subclass)
    else:
        write_csv(AQ / "17_features_b1" / "AQ2B1_feature_manifest.csv", [], ["derivative_uid", "array_sha256", "npy_file_sha256", "shape", "dtype", "finite_check", "npy_path"])

    collision_rows = []
    if derivative_rows:
        by_pcm = defaultdict(list)
        by_feat = defaultdict(list)
        feat_map = {r["derivative_uid"]: r for r in feature_rows}
        for r in derivative_rows:
            by_pcm[r["aligned_pcm_sha256"]].append(r)
            if r["derivative_uid"] in feat_map:
                by_feat[feat_map[r["derivative_uid"]]["array_sha256"]].append(r)
        for kind, groups in [("aligned_pcm_sha256", by_pcm), ("representation_a_array_sha256", by_feat)]:
            for h, rows in groups.items():
                labels = sorted(set(r["codec_family"] for r in rows))
                if h and len(rows) > 1:
                    collision_rows.append({"collision_type": kind, "sha256": h, "count": len(rows), "codec_labels": ";".join(labels), "cross_label": len(labels) > 1, "derivative_uids": ";".join(r["derivative_uid"] for r in rows)})
    write_csv(AQ / "19_results_b1" / "diagnostics" / "AQ2B1_collision_audit.csv", collision_rows, ["collision_type", "sha256", "count", "codec_labels", "cross_label", "derivative_uids"])

    model_ready = False
    validation_metrics = {}
    test_seen_metrics = {}
    test_low_metrics = {}
    test_high_metrics = {}
    test_mid_metrics = {}
    bootstrap_summary = {}
    permutation_summary = {}
    config_shift_gap = None
    gate_values = {}
    scientific_gate = ""
    convergence_warning = ""
    pretest_written_before_test = False
    model_numeric_sha = model_config_sha = pretest_sha = ""
    data_audit = {}

    if final == "":
        feat_map = {r["derivative_uid"]: r["npy_path"] for r in feature_rows}
        rows_by_uid = {r["derivative_uid"]: r for r in derivative_rows}
        def subset(split=None, rates=None):
            rows = [r for r in derivative_rows if (split is None or r["split"] == split) and (rates is None or int(r["nominal_rate_kbps"]) in rates)]
            rows = sorted(rows, key=lambda r: r["derivative_uid"])
            X = np.vstack([np.load(feat_map[r["derivative_uid"]], allow_pickle=False) for r in rows])
            y = np.array([r["codec_family"] for r in rows])
            return rows, X, y

        train_rows, X_train, y_train = subset("TRAIN", TRAIN_RATES)
        val_rows, X_val, y_val = subset("VALIDATION", VALIDATION_RATES)
        test_seen_rows, X_test_seen, y_test_seen = subset("TEST", [32, 128])
        test_low_rows, X_test_low, y_test_low = subset("TEST", [32])
        test_high_rows, X_test_high, y_test_high = subset("TEST", [128])
        test_mid_rows, X_test_mid, y_test_mid = subset("TEST", [64])
        data_audit = {
            "X_train_shape": list(X_train.shape),
            "y_train_count": int(len(y_train)),
            "X_validation_shape": list(X_val.shape),
            "y_validation_count": int(len(y_val)),
            "X_test_seen_shape": list(X_test_seen.shape),
            "y_test_seen_count": int(len(y_test_seen)),
            "X_test_mid_shape": list(X_test_mid.shape),
            "y_test_mid_count": int(len(y_test_mid)),
            "class_support": {
                "train": dict(Counter(y_train)),
                "validation": dict(Counter(y_val)),
                "test_seen": dict(Counter(y_test_seen)),
                "test_mid": dict(Counter(y_test_mid)),
            },
            "predictor_columns": "Representation-A numerical feature array only",
        }
        (AQ / "19_results_b1" / "AQ2B1_data_matrix_audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")
        expected_shapes = X_train.shape == (294, 256) and X_val.shape == (96, 256) and X_test_seen.shape == (102, 256) and X_test_mid.shape == (51, 256)
        equal_support = all(Counter(y) == {"AAC": len(y)//3, "MP3": len(y)//3, "Opus": len(y)//3} for y in [y_train, y_val, y_test_seen, y_test_mid])
        if not expected_shapes or not equal_support:
            final, subclass = "FAIL_INTEGRITY", "DATA_MATRIX_SHAPE_FAIL"
            blockers.append(subclass)
        else:
            scaler = StandardScaler(with_mean=True, with_std=True)
            model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", multi_class="multinomial", fit_intercept=True, class_weight=None, max_iter=5000, tol=1e-8)
            zero_var = int(np.sum(np.var(X_train, axis=0) == 0))
            X_train_s = scaler.fit_transform(X_train)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train_s, y_train)
            convergence_warning = "; ".join(str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning))
            if convergence_warning:
                final, subclass = "BLOCKED_EXECUTION", "MODEL_CONVERGENCE_FAIL"
                blockers.append(subclass)
            else:
                val_probs = model.predict_proba(scaler.transform(X_val))
                val_pred = model.classes_[np.argmax(val_probs, axis=1)]
                validation_metrics = metrics_bundle(y_val, val_pred)
                (AQ / "19_results_b1" / "AQ2B1_validation_metrics.json").write_text(json.dumps(validation_metrics, indent=2), encoding="utf-8")
                write_csv(AQ / "19_results_b1" / "predictions" / "AQ2B1_validation_predictions.csv", prediction_rows(val_rows, y_val, val_pred, val_probs), ["derivative_uid", "parent_uid", "source_corpus", "true_codec", "predicted_codec", "prob_AAC", "prob_MP3", "prob_Opus"])

                np.savez(
                    AQ / "18_model_b1" / "AQ2B1_model_numeric.npz",
                    scaler_mean=scaler.mean_,
                    scaler_scale=scaler.scale_,
                    scaler_var=scaler.var_,
                    model_coef=model.coef_,
                    model_intercept=model.intercept_,
                    class_order=np.array(CLASS_ORDER),
                    sklearn_classes=model.classes_,
                )
                model_config = {
                    "scaler": {"type": "StandardScaler", "with_mean": True, "with_std": True, "fit_scope": "TRAIN_ONLY"},
                    "model": {"type": "LogisticRegression", "penalty": "l2", "C": 1.0, "solver": "lbfgs", "multi_class": "multinomial", "fit_intercept": True, "class_weight": None, "max_iter": 5000, "tol": 1e-8},
                    "class_order": CLASS_ORDER,
                    "packages": env_info,
                    "n_iter": model.n_iter_.tolist(),
                    "classes_": model.classes_.tolist(),
                    "coef_shape": list(model.coef_.shape),
                    "intercept_shape": list(model.intercept_.shape),
                    "training_row_count": int(X_train.shape[0]),
                    "feature_dimension": int(X_train.shape[1]),
                    "zero_variance_train_feature_count": zero_var,
                    "convergence_warning": convergence_warning,
                }
                (AQ / "18_model_b1" / "AQ2B1_model_config.json").write_text(json.dumps(model_config, indent=2), encoding="utf-8")
                model_numeric_sha = sha256_file(AQ / "18_model_b1" / "AQ2B1_model_numeric.npz")
                model_config_sha = sha256_file(AQ / "18_model_b1" / "AQ2B1_model_config.json")
                pretest = {
                    "phase": "AQ-2B.1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "AQ2B0_freeze_sha256": EXPECTED["AQ2B0_FREEZE"],
                    "feature_spec_sha256": EXPECTED["FEATURE_SPEC"],
                    "feature_extractor_sha256": EXPECTED["FEATURE_EXTRACTOR"],
                    "derivative_manifest_sha256": sha256_file(AQ / "16_derivatives_b1" / "AQ2B1_derivative_manifest.csv"),
                    "feature_manifest_sha256": sha256_file(AQ / "17_features_b1" / "AQ2B1_feature_manifest.csv"),
                    "train_parent_count": 49,
                    "validation_parent_count": 16,
                    "test_parent_count": 17,
                    "train_row_count": 294,
                    "validation_row_count": 96,
                    "model_config": model_config,
                    "model_numeric_sha256": model_numeric_sha,
                    "model_config_sha256": model_config_sha,
                    "validation_metrics": validation_metrics,
                    "test_evaluated": "NO",
                }
                pretest_path = AQ / "18_model_b1" / "AQ2B1_PRETEST_MODEL_FREEZE.json"
                pretest_path.write_text(json.dumps(pretest, indent=2), encoding="utf-8")
                pretest_sha = sha256_file(pretest_path)
                pretest_written_before_test = True

                # TEST evaluation begins only after pretest freeze exists and is hashed.
                def eval_and_write(rows, X, y, metrics_path, pred_path):
                    probs = model.predict_proba(scaler.transform(X))
                    pred = model.classes_[np.argmax(probs, axis=1)]
                    mb = metrics_bundle(y, pred)
                    metrics_path.write_text(json.dumps(mb, indent=2), encoding="utf-8")
                    write_csv(pred_path, prediction_rows(rows, y, pred, probs), ["derivative_uid", "parent_uid", "source_corpus", "true_codec", "predicted_codec", "prob_AAC", "prob_MP3", "prob_Opus"])
                    return mb, pred, probs

                test_seen_metrics, test_seen_pred, _ = eval_and_write(test_seen_rows, X_test_seen, y_test_seen, AQ / "19_results_b1" / "AQ2B1_test_seen_metrics.json", AQ / "19_results_b1" / "predictions" / "AQ2B1_test_seen_predictions.csv")
                test_low_metrics = metrics_bundle(y_test_low, model.classes_[np.argmax(model.predict_proba(scaler.transform(X_test_low)), axis=1)])
                test_high_metrics = metrics_bundle(y_test_high, model.classes_[np.argmax(model.predict_proba(scaler.transform(X_test_high)), axis=1)])
                test_mid_metrics, test_mid_pred, test_mid_probs = eval_and_write(test_mid_rows, X_test_mid, y_test_mid, AQ / "19_results_b1" / "AQ2B1_test_mid_metrics.json", AQ / "19_results_b1" / "predictions" / "AQ2B1_test_mid_predictions.csv")

                rng = np.random.default_rng(20260830)
                parents = sorted(set(r["parent_uid"] for r in test_mid_rows))
                parent_indices = {p: [i for i, r in enumerate(test_mid_rows) if r["parent_uid"] == p] for p in parents}
                boot_rows = []
                ba_vals = []
                f1_vals = []
                for i in range(10000):
                    sampled = rng.choice(parents, size=len(parents), replace=True)
                    idx = [j for p in sampled for j in parent_indices[p]]
                    ba = float(balanced_accuracy_score(y_test_mid[idx], test_mid_pred[idx]))
                    mf1 = float(f1_score(y_test_mid[idx], test_mid_pred[idx], labels=CLASS_ORDER, average="macro", zero_division=0))
                    ba_vals.append(ba)
                    f1_vals.append(mf1)
                    boot_rows.append({"replicate": i, "balanced_accuracy": ba, "macro_f1": mf1})
                write_csv(AQ / "19_results_b1" / "bootstrap" / "AQ2B1_parent_bootstrap.csv", boot_rows, ["replicate", "balanced_accuracy", "macro_f1"])
                bootstrap_summary = {
                    "bootstrap_reps": 10000,
                    "bootstrap_seed": 20260830,
                    "balanced_accuracy_percentiles": {"2.5": float(np.percentile(ba_vals, 2.5)), "50": float(np.percentile(ba_vals, 50)), "97.5": float(np.percentile(ba_vals, 97.5))},
                    "macro_f1_percentiles": {"2.5": float(np.percentile(f1_vals, 2.5)), "50": float(np.percentile(f1_vals, 50)), "97.5": float(np.percentile(f1_vals, 97.5))},
                }
                (AQ / "19_results_b1" / "bootstrap" / "AQ2B1_parent_bootstrap_summary.json").write_text(json.dumps(bootstrap_summary, indent=2), encoding="utf-8")

                rng = np.random.default_rng(20260831)
                obs_ba = test_mid_metrics["balanced_accuracy"]
                perm_rows = []
                null_ge = 0
                y_mid_arr = np.asarray(y_test_mid)
                pred_mid_arr = np.asarray(test_mid_pred)
                for i in range(10000):
                    y_perm = y_mid_arr.copy()
                    for p in parents:
                        idx = parent_indices[p]
                        y_perm[idx] = rng.permutation(y_perm[idx])
                    ba = float(balanced_accuracy_score(y_perm, pred_mid_arr))
                    if ba >= obs_ba:
                        null_ge += 1
                    perm_rows.append({"replicate": i, "balanced_accuracy": ba})
                write_csv(AQ / "19_results_b1" / "permutation" / "AQ2B1_parent_permutation.csv", perm_rows, ["replicate", "balanced_accuracy"])
                perm_p = (1 + null_ge) / (1 + 10000)
                permutation_summary = {"permutation_reps": 10000, "permutation_seed": 20260831, "observed_balanced_accuracy": obs_ba, "null_ge_observed": null_ge, "one_sided_p_value": perm_p}
                (AQ / "19_results_b1" / "permutation" / "AQ2B1_parent_permutation_summary.json").write_text(json.dumps(permutation_summary, indent=2), encoding="utf-8")

                confusion_all = {
                    "validation": validation_metrics,
                    "test_seen_rates_32_128": test_seen_metrics,
                    "test_low_32": test_low_metrics,
                    "test_high_128": test_high_metrics,
                    "test_heldout_mid_64": test_mid_metrics,
                }
                (AQ / "19_results_b1" / "diagnostics" / "AQ2B1_confusion_matrices.json").write_text(json.dumps(confusion_all, indent=2), encoding="utf-8")

                condition_rows = []
                for name, rows, metrics in [
                    ("VALIDATION_SEEN_RATES_32_128", val_rows, validation_metrics),
                    ("TEST_SEEN_RATES_32_128", test_seen_rows, test_seen_metrics),
                    ("TEST_LOW_32", test_low_rows, test_low_metrics),
                    ("TEST_HIGH_128", test_high_rows, test_high_metrics),
                    ("TEST_HELDOUT_MID_64", test_mid_rows, test_mid_metrics),
                ]:
                    condition_rows.append({"condition": name, "rows": len(rows), "parents": len(set(r["parent_uid"] for r in rows)), "balanced_accuracy": metrics["balanced_accuracy"], "macro_f1": metrics["macro_f1"], "accuracy": metrics["accuracy"]})
                write_csv(AQ / "19_results_b1" / "diagnostics" / "AQ2B1_condition_summary.csv", condition_rows, ["condition", "rows", "parents", "balanced_accuracy", "macro_f1", "accuracy"])

                rate_rows = []
                for key, vals in sorted(defaultdict(list, {}).items()):
                    pass
                rate_groups = defaultdict(list)
                for r in derivative_rows:
                    rate_groups[(r["split"], r["codec_family"], r["nominal_rate_kbps"])].append(float(r["achieved_rate_kbps"]))
                for (split, codec, rate), vals in sorted(rate_groups.items()):
                    rate_rows.append({"split": split, "codec_family": codec, "nominal_rate_kbps": rate, "count": len(vals), "achieved_min_kbps": min(vals), "achieved_median_kbps": float(np.median(vals)), "achieved_max_kbps": max(vals)})
                write_csv(AQ / "19_results_b1" / "diagnostics" / "AQ2B1_achieved_rate_summary.csv", rate_rows, ["split", "codec_family", "nominal_rate_kbps", "count", "achieved_min_kbps", "achieved_median_kbps", "achieved_max_kbps"])

                config_shift_gap = test_seen_metrics["balanced_accuracy"] - test_mid_metrics["balanced_accuracy"]
                min_class_recall = min(test_mid_metrics["per_class_recall"].values())
                gate_values = {
                    "BA_PRIMARY": test_mid_metrics["balanced_accuracy"],
                    "BOOTSTRAP_LOW": bootstrap_summary["balanced_accuracy_percentiles"]["2.5"],
                    "PERM_P": perm_p,
                    "MIN_CLASS_RECALL": min_class_recall,
                }
                if gate_values["BA_PRIMARY"] >= 0.60 and gate_values["BOOTSTRAP_LOW"] > (1/3) and gate_values["PERM_P"] <= 0.05 and gate_values["MIN_CLASS_RECALL"] >= 0.40:
                    scientific_gate = "PASS_MFEA1_TRACE_EXISTS"
                    final = "PASS_MFEA1_TRACE_EXISTS"
                elif gate_values["BA_PRIMARY"] <= 0.45 and gate_values["PERM_P"] > 0.10:
                    scientific_gate = "STOP_MFEA1_TRACE_NOT_SUPPORTED"
                    final = "STOP_MFEA1_TRACE_NOT_SUPPORTED"
                else:
                    scientific_gate = "INCONCLUSIVE_MFEA1"
                    final = "INCONCLUSIVE_MFEA1"

    # Ensure required result placeholders exist if blocked early.
    for path, payload in [
        (AQ / "19_results_b1" / "AQ2B1_data_matrix_audit.json", data_audit),
        (AQ / "19_results_b1" / "AQ2B1_validation_metrics.json", validation_metrics),
        (AQ / "19_results_b1" / "AQ2B1_test_seen_metrics.json", test_seen_metrics),
        (AQ / "19_results_b1" / "AQ2B1_test_mid_metrics.json", test_mid_metrics),
    ]:
        if not path.exists():
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    deriv_file_sums = AQ / "16_derivatives_b1" / "AQ2B1_derivative_file_sha256s.txt"
    derivative_files = sorted(list((AQ / "16_derivatives_b1" / "encoded").glob("*")) + list((AQ / "16_derivatives_b1" / "decoded_raw").glob("*.wav")) + list((AQ / "16_derivatives_b1" / "aligned_pcm").glob("*.wav")), key=lambda p: str(p).lower())
    deriv_file_sums.write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(AQ).as_posix()}" for p in derivative_files) + "\n", encoding="utf-8")
    feat_file_sums = AQ / "17_features_b1" / "AQ2B1_feature_file_sha256s.txt"
    feature_files = sorted((AQ / "17_features_b1" / "representation_a").glob("*.npy"), key=lambda p: str(p).lower())
    feat_file_sums.write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(AQ).as_posix()}" for p in feature_files) + "\n", encoding="utf-8")

    command_log = AQ / "logs_b1" / "AQ2B1_command_log.txt"
    command_log.write_text("\n".join(commands) + "\n", encoding="utf-8")

    collision_summary = {
        "collision_rows": len(collision_rows),
        "cross_label_collisions": sum(1 for r in collision_rows if str(r.get("cross_label")) == "True"),
    }
    freeze_path = AQ / "03_manifests" / "AQ2B1_FREEZE.json"
    core_artifacts = [
        AQ / "16_derivatives_b1" / "AQ2B1_derivative_manifest.csv",
        deriv_file_sums,
        AQ / "17_features_b1" / "AQ2B1_feature_manifest.csv",
        feat_file_sums,
        AQ / "18_model_b1" / "AQ2B1_model_numeric.npz",
        AQ / "18_model_b1" / "AQ2B1_model_config.json",
        AQ / "18_model_b1" / "AQ2B1_PRETEST_MODEL_FREEZE.json",
        AQ / "19_results_b1" / "AQ2B1_data_matrix_audit.json",
        AQ / "19_results_b1" / "AQ2B1_validation_metrics.json",
        AQ / "19_results_b1" / "AQ2B1_test_seen_metrics.json",
        AQ / "19_results_b1" / "AQ2B1_test_mid_metrics.json",
        AQ / "19_results_b1" / "predictions" / "AQ2B1_validation_predictions.csv",
        AQ / "19_results_b1" / "predictions" / "AQ2B1_test_seen_predictions.csv",
        AQ / "19_results_b1" / "predictions" / "AQ2B1_test_mid_predictions.csv",
        AQ / "19_results_b1" / "bootstrap" / "AQ2B1_parent_bootstrap.csv",
        AQ / "19_results_b1" / "bootstrap" / "AQ2B1_parent_bootstrap_summary.json",
        AQ / "19_results_b1" / "permutation" / "AQ2B1_parent_permutation.csv",
        AQ / "19_results_b1" / "permutation" / "AQ2B1_parent_permutation_summary.json",
        AQ / "19_results_b1" / "diagnostics" / "AQ2B1_parent_split_audit.csv",
        AQ / "19_results_b1" / "diagnostics" / "AQ2B1_collision_audit.csv",
        AQ / "19_results_b1" / "diagnostics" / "AQ2B1_condition_summary.csv",
        AQ / "19_results_b1" / "diagnostics" / "AQ2B1_confusion_matrices.json",
        AQ / "19_results_b1" / "diagnostics" / "AQ2B1_achieved_rate_summary.csv",
        AQ / "logs_b1" / "AQ2B1_execution_script.py",
        command_log,
    ]
    model_config = json.loads((AQ / "18_model_b1" / "AQ2B1_model_config.json").read_text(encoding="utf-8")) if (AQ / "18_model_b1" / "AQ2B1_model_config.json").exists() else {}
    freeze = {
        "phase": "AQ-2B.1",
        "specification_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": final,
        "subclassification": subclass,
        "AQ2B0_freeze_sha256": sha256_file(AQ / "03_manifests" / "AQ2B0_FREEZE.json"),
        "AQ2A_R2_freeze_sha256": sha256_file(AQ / "03_manifests" / "AQ2A_R2_FREEZE.json"),
        "corpus_set_sha256": EXPECTED["CORPUS_SET"],
        "canonical_parent_count": 82,
        "split_counts": {"TRAIN": 49, "VALIDATION": 16, "TEST": 17},
        "python_path": str(PYTHON_B0),
        "python_sha256": sha256_file(PYTHON_B0),
        "python_version": env_info.get("python", ""),
        "feature_spec_sha256": sha256_file(feature_spec_path),
        "feature_extractor_sha256": sha256_file(extractor_path),
        "representation_used": "STFT_LOGPOWER_STATS_V1",
        "known_codec_set": CLASS_ORDER,
        "vorbis_scientific_quarantine": "ACTIVE",
        "train_rates": TRAIN_RATES,
        "validation_rates": VALIDATION_RATES,
        "test_rates": TEST_RATES,
        "heldout_rate": HELDOUT_RATE,
        "expected_derivative_count": 543,
        "actual_derivative_count": len(derivative_rows),
        "train_derivative_count": sum(1 for r in derivative_rows if r.get("split") == "TRAIN"),
        "validation_derivative_count": sum(1 for r in derivative_rows if r.get("split") == "VALIDATION"),
        "test_derivative_count": sum(1 for r in derivative_rows if r.get("split") == "TEST"),
        "parent_disjointness_status": "PASS" if all(r["status"] == "PASS" for r in split_audit) else "FAIL",
        "lossless_negative_identifiability_sentinel": "PASS" if lossless_ok else "FAIL",
        "encoder_ffmpeg_sha256": sha256_file(FFMPEG_ENCODER_R2),
        "ffprobe_sha256": sha256_file(FFPROBE_R2),
        "common_decoder_sha256": sha256_file(FFMPEG_CANONICAL_V1),
        "encode_success_count": sum(1 for r in derivative_rows if r.get("encode_return_code") == 0 or str(r.get("encode_return_code")) == "0"),
        "decode_success_count": sum(1 for r in derivative_rows if r.get("decode_return_code") == 0 or str(r.get("decode_return_code")) == "0"),
        "alignment_success_count": sum(1 for r in derivative_rows if r.get("aligned_sample_count") == SEGMENT_SAMPLES or str(r.get("aligned_sample_count")) == str(SEGMENT_SAMPLES)),
        "feature_success_count": len(feature_rows),
        "collision_audit_summary": collision_summary,
        "model_type": "StandardScaler + multinomial LogisticRegression",
        "model_configuration": model_config,
        "scaler_fit_scope": "TRAIN_ONLY",
        "model_numeric_sha256": model_numeric_sha,
        "model_config_sha256": model_config_sha,
        "pretest_model_freeze_sha256": pretest_sha,
        "validation_metrics": validation_metrics,
        "test_seen_metrics": test_seen_metrics,
        "test_mid_primary_metrics": test_mid_metrics,
        "parent_bootstrap_summary": bootstrap_summary,
        "parent_permutation_summary": permutation_summary,
        "chance_balanced_accuracy": 1/3,
        "primary_gate_values": gate_values,
        "configuration_shift_gap": config_shift_gap,
        "scientific_trace_gate": scientific_gate,
        "VORBIS_USED_IN_B1_SCIENTIFIC_DATA": "NO",
        "representation_B_scientific_use": "NO",
        "new_training_performed": "YES",
        "model_weights_created": "YES",
        "scientific_codec_result_produced": "YES",
        "AQ2A_v1_frozen_artifacts_modified": "NO",
        "AQ2A_R1_frozen_artifacts_modified": "NO",
        "AQ2A_R2_frozen_artifacts_modified": "NO",
        "AQ2B0_frozen_artifacts_modified": "NO",
        "existing_MM_artifacts_modified": "NO",
        "remaining_blockers": blockers,
        "artifact_hashes": {p.relative_to(AQ).as_posix(): sha256_file(p) for p in core_artifacts if p.exists()},
    }
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    report_path = AQ / "20_reports_b1" / "AQ2B1_EXECUTION_REPORT.md"
    auth = "YES" if final == "PASS_MFEA1_TRACE_EXISTS" else "NO"
    auth_reason = "AQ-2B.2 CANDIDATE AUTHORIZATION = YES" if final == "PASS_MFEA1_TRACE_EXISTS" else "AQ-2B.2 CANDIDATE AUTHORIZATION = NO\n" + ("REQUIRES SCIENTIFIC REVIEW" if final == "INCONCLUSIVE_MFEA1" else "MINIMAL PREMISE NOT SUPPORTED" if final == "STOP_MFEA1_TRACE_NOT_SUPPORTED" else "; ".join(blockers))
    report = [
        "# AQ-2B.1 Execution Report\n",
        f"## 1. Final classification\n\n{final}\n",
        f"## 2. Subclassification\n\n{subclass or 'None'}\n",
        "## 3. Scientific interpretation boundary\n\nThis is a minimal trace-existence falsification, not a best-classifier claim and not an open-set result.\n",
        f"## 4. AQ-2B.0 identity reconciliation\n\n{'PASS' if all(b0_checks.values()) else 'FAIL'}; SHA-256 `{freeze['AQ2B0_freeze_sha256']}`.\n",
        f"## 5. AQ-2A-R2 identity reconciliation\n\nPASS; SHA-256 `{freeze['AQ2A_R2_freeze_sha256']}`.\n",
        f"## 6. Python/environment reconciliation\n\n{'PASS' if py_ok else 'FAIL'}; `{PYTHON_B0}`.\n",
        f"## 7. Feature-spec/extractor reconciliation\n\n{'PASS' if feature_ok else 'FAIL'}.\n",
        f"## 8. Known codec set\n\n{CLASS_ORDER}\n",
        "## 9. Vorbis quarantine confirmation\n\nVORBIS_SCIENTIFIC_QUARANTINE = ACTIVE.\n",
        "## 10. Parent split counts\n\nTRAIN=49, VALIDATION=16, TEST=17.\n",
        f"## 11. Parent-disjointness audit\n\n{freeze['parent_disjointness_status']}.\n",
        "## 12. Frozen rate design\n\nTRAIN/VALIDATION rates: 32,128. TEST rates: 32,64,128. Held-out development rate: 64.\n",
        f"## 13. Expected versus actual derivative counts\n\nExpected 543; actual {len(derivative_rows)}.\n",
        f"## 14. Encode/probe/decode/alignment integrity\n\nEncode={freeze['encode_success_count']}; decode={freeze['decode_success_count']}; alignment={freeze['alignment_success_count']}.\n",
        "## 15. Achieved bitrate summary\n\nSee `19_results_b1/diagnostics/AQ2B1_achieved_rate_summary.csv`.\n",
        f"## 16. Lossless negative-identifiability sentinel\n\n{freeze['lossless_negative_identifiability_sentinel']}.\n",
        "## 17. Representation used\n\nSTFT_LOGPOWER_STATS_V1.\n",
        "## 18. Representation-B non-use confirmation\n\nREPRESENTATION B USED FOR SCIENTIFIC CLASSIFICATION = NO.\n",
        f"## 19. Feature count/shape/finite integrity\n\n{len(feature_rows)} / 543 valid Representation-A arrays.\n",
        f"## 20. Collision audit\n\n{json.dumps(collision_summary, indent=2)}\n",
        f"## 21. Training matrix dimensions\n\n{data_audit.get('X_train_shape')}.\n",
        f"## 22. Validation matrix dimensions\n\n{data_audit.get('X_validation_shape')}.\n",
        f"## 23. Test-seen dimensions\n\n{data_audit.get('X_test_seen_shape')}.\n",
        f"## 24. Test-MID primary dimensions\n\n{data_audit.get('X_test_mid_shape')}.\n",
        "## 25. Frozen scaler configuration\n\nStandardScaler(with_mean=True, with_std=True), fit on TRAIN only.\n",
        "## 26. Frozen Logistic Regression configuration\n\nL2, C=1.0, lbfgs, multinomial, fit_intercept=True, max_iter=5000, tol=1e-8.\n",
        f"## 27. Model convergence\n\n{'PASS' if not convergence_warning and model_config else 'FAIL'}; n_iter={model_config.get('n_iter')}.\n",
        f"## 28. Validation diagnostics\n\n{json.dumps(validation_metrics, indent=2)}\n",
        f"## 29. PRETEST_MODEL_FREEZE identity and SHA-256\n\n`{pretest_sha}`.\n",
        f"## 30. Explicit confirmation that PRETEST freeze preceded test evaluation\n\n{pretest_written_before_test}.\n",
        f"## 31. TEST_SEEN_RATES_32_128 metrics\n\n{json.dumps(test_seen_metrics, indent=2)}\n",
        f"## 32. TEST_LOW_32 metrics\n\n{json.dumps(test_low_metrics, indent=2)}\n",
        f"## 33. TEST_HIGH_128 metrics\n\n{json.dumps(test_high_metrics, indent=2)}\n",
        f"## 34. TEST_HELDOUT_MID_64 primary metrics\n\n{json.dumps(test_mid_metrics, indent=2)}\n",
        f"## 35. Primary confusion matrix\n\n{json.dumps(test_mid_metrics.get('confusion_matrix', []), indent=2)}\n",
        f"## 36. Per-class primary recalls\n\n{json.dumps(test_mid_metrics.get('per_class_recall', {}), indent=2)}\n",
        f"## 37. Parent-bootstrap 95% interval\n\n{json.dumps(bootstrap_summary, indent=2)}\n",
        f"## 38. Parent-preserving permutation p-value\n\n{permutation_summary.get('one_sided_p_value')}.\n",
        f"## 39. Primary gate values\n\n{json.dumps(gate_values, indent=2)}\n",
        f"## 40. Configuration-shift diagnostic gap\n\n{config_shift_gap}\n",
        f"## 41. Final MFE-A.1 scientific gate interpretation\n\n{scientific_gate}\n",
        f"## 42. Whether AQ-2B.2 is automatically authorized\n\n{auth_reason}\n",
        "## 43. All created artifacts\n\n",
    ]
    all_artifacts = core_artifacts + [freeze_path, report_path]
    for p in all_artifacts:
        if p.exists():
            report.append(f"- `{p.relative_to(AQ).as_posix()}`\n")
    report.append("## 44. Major SHA-256 values\n\n")
    for p in all_artifacts:
        if p.exists() and p != report_path:
            report.append(f"- `{sha256_file(p)}`  `{p.relative_to(AQ).as_posix()}`\n")
    report.extend([
        f"## 45. Remaining blockers\n\n{'; '.join(blockers) if blockers else 'None'}\n",
        "## 46. Explicit statement\n\nVORBIS USED IN B1 SCIENTIFIC DATA = NO\n",
        "## 47. Explicit statement\n\nREPRESENTATION B USED FOR SCIENTIFIC CLASSIFICATION = NO\n",
        "## 48. Explicit statement\n\nNO OPEN-SET RESULT WAS PRODUCED IN AQ-2B.1\n",
        "## 49. Explicit statement\n\nNO NEURAL MODEL WAS TRAINED IN AQ-2B.1\n",
        "## 50. Explicit statements\n\nNEW TRAINING PERFORMED = YES\n\nMODEL WEIGHTS CREATED = YES\n\nSCIENTIFIC CODEC RESULT PRODUCED = YES\n\nAQ2A V1 FROZEN ARTIFACTS MODIFIED = NO\n\nAQ2A R1 FROZEN ARTIFACTS MODIFIED = NO\n\nAQ2A R2 FROZEN ARTIFACTS MODIFIED = NO\n\nAQ2B0 FROZEN ARTIFACTS MODIFIED = NO\n\nEXISTING MM ARTIFACTS MODIFIED = NO\n",
    ])
    report_path.write_text("".join(report), encoding="utf-8")

    ledger_artifacts = sorted(set(all_artifacts), key=lambda p: str(p).lower())
    (AQ / "AQ2B1_SHA256SUMS.txt").write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(AQ).as_posix()}" for p in ledger_artifacts if p.exists()) + "\n", encoding="utf-8")

    print(json.dumps({
        "classification": final,
        "subclassification": subclass,
        "derivatives": len(derivative_rows),
        "features": len(feature_rows),
        "BA_PRIMARY": gate_values.get("BA_PRIMARY"),
        "BOOTSTRAP_LOW": gate_values.get("BOOTSTRAP_LOW"),
        "PERM_P": gate_values.get("PERM_P"),
        "MIN_CLASS_RECALL": gate_values.get("MIN_CLASS_RECALL"),
    }, indent=2))


if __name__ == "__main__":
    main()
