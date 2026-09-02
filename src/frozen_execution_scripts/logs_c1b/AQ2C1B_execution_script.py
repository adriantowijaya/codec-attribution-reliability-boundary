import csv
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import correlate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
import sklearn


ROOT = Path(r"[REDACTED_LOCAL_PATH]\Tony Hidayat\AQ_OpenWorld_Codec_Provenance")
PYTHON = Path(r"[REDACTED_LOCAL_PATH]\anaconda3\python.exe")
CLASS_ORDER = ["AAC", "MP3", "Opus"]
CORPORA = ["Orchset", "IRMAS", "FSDnoisy18k"]
RATES = [32, 48, 64, 80, 96, 128]
NEW_RATES = [48, 80, 96]
BOOT_RATES = [48, 64, 80, 96]
SEGMENT_SAMPLES = 96000
SEGMENT_SECONDS = 2.0
ALIGN_BOUND = 8192
EXPECTED = {
    "AQ2C1A_FREEZE": "89c5810d538f60bc588ca54fa7f81e6dcdcab81e2509219cebeae7104b407fb9",
    "PROSPECTIVE_SPEC": "cc65c5b616e03106dbccbadeb2be1a36bf0f66ecdd67700dc7e94ac3f9cf1a5f",
    "CONFIRM_PARENT_MANIFEST": "8f5cace862a0066e7bc6c47e23816206dc9069d8841234186c20a747868e69d8",
    "CONFIRM_PCM_HASHES": "f03a621d43c871ceb3e12273d9f91bf23a7946f0b4ddf191270f6216dcb00fca",
    "DISCOVERY_EXCLUSION": "0ee6c0cfb62bfcfd98871af6e1d486e06caa4d578dab723abfec03a4f94e3395",
    "MODEL_NUMERIC": "7850e14a7231ac5d2d06a72c6c2eb487cedd7df32cf5750bf62ddda79932b5de",
    "MODEL_CONFIG": "495b976b5fe558faee6a374707e5629c04a8d8b844f5aea9d178259db5d1fdff",
    "PRETEST": "2eb6c5216196a72466693d7b68a919762d1ebd4b8cb5207ee7639b8758a67f2c",
    "FEATURE_SPEC": "83020402082b7168d88df8d1eb02fc4f254d24b6fbd4ab0770403d1f96c1acae",
    "FEATURE_EXTRACTOR": "6c79cbc06c67c9cff140cd64de6f878ea59f1b1580bb7db0b0acb08a4a57da8f",
    "PYTHON": "62c225fb9cdc41b139c7024581c233644f975ffc35314558c60ebefa6b88be01",
    "ENCODER": "57c56e369d5b4873b4d93fc1a1d833cb7cd8bc9325c14b05c34ce60b22842d8a",
    "FFPROBE": "afe05347caaabe479b3c4eae71992b6ec1e11c57266a1d665deb0f9fe9847208",
    "DECODER": "4dc3e63209cb6f183b703c8842f6e3dcc22778ccca1a3b9f4b5fca4034bb54dd",
    "B1_DERIVATIVE_MANIFEST": "6f32143b4e0267e3f2acdf91404954a9ddb1e2ad5c5d5bf6cb99651dc9e021aa",
    "B1_FEATURE_MANIFEST": "e6e9983484bea4d44756a7b9b71bbea85a687fbd07892cdd42ff9fcbbec74754",
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
        data = w.readframes(w.getnframes())
    return hashlib.sha256(data).hexdigest()


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
    samples = np.asarray(samples, dtype="<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def load_extractor():
    path = ROOT / "14_feature_preflight_b0" / "aq2b0_feature_extractor.py"
    spec = importlib.util.spec_from_file_location("aq2b0_feature_extractor", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def array_hash(arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    payload = b"STFT_LOGPOWER_STATS_V1\n" + ",".join(str(x) for x in arr.shape).encode("ascii") + b"\n<f8\n" + arr.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def codec_info(codec, rate):
    if codec == "AAC":
        return {"encoder": "aac", "core": "FFmpeg-native AAC", "core_uid": "FFmpeg-native AAC", "ext": "m4a", "probe": "aac", "args": ["-c:a", "aac", "-b:a", f"{rate}k"]}
    if codec == "MP3":
        return {"encoder": "libmp3lame", "core": "LAME", "core_uid": "LAME", "ext": "mp3", "probe": "mp3", "args": ["-c:a", "libmp3lame", "-b:a", f"{rate}k"]}
    if codec == "Opus":
        return {"encoder": "libopus", "core": "libopus", "core_uid": "libopus", "ext": "opus", "probe": "opus", "args": ["-c:a", "libopus", "-b:a", f"{rate}k", "-vbr", "off"]}
    raise ValueError(codec)


def stable_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def summary_stats(vals):
    vals = np.asarray(list(vals), dtype=np.float64)
    if vals.size == 0:
        return {"median": None, "iqr": None, "min": None, "max": None}
    return {"median": float(np.median(vals)), "iqr": float(np.percentile(vals, 75) - np.percentile(vals, 25)), "min": float(np.min(vals)), "max": float(np.max(vals))}


def metrics_for(rows):
    y_true = [r["true_codec"] for r in rows]
    y_pred = [r["predicted_codec"] for r in rows]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_codec_recall": {c: float(recall_score(y_true, y_pred, labels=[c], average="macro", zero_division=0)) for c in CLASS_ORDER},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_ORDER).tolist(),
        "class_order": CLASS_ORDER,
        "n": len(rows),
    }


def dominant_error(items, codec):
    counts = Counter(r["predicted_codec"] for r in items)
    correct = counts.get(codec, 0)
    errors = len(items) - correct
    wrong_counts = {c: counts.get(c, 0) for c in CLASS_ORDER if c != codec}
    if errors == 0:
        return "NA", None
    target, count = max(wrong_counts.items(), key=lambda kv: kv[1])
    return target, float(count / errors)


def fail(subclassification, detail):
    freeze = {
        "phase": "AQ-2C.1B",
        "specification_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": "FAIL_INTEGRITY",
        "subclassification": subclassification,
        "detail": detail,
        "prediction_count": 0,
        "model_refit_performed": "NO",
        "new_training_performed": "NO",
        "new_model_weights_created": "NO",
        "scientific_confirmation_result_produced": "NO",
        "vorbis_used": "NO",
        "representation_B_used": "NO",
        "remaining_blockers": [],
    }
    write_json(ROOT / "03_manifests" / "AQ2C1B_FREEZE.json", freeze)
    raise SystemExit(2)


def main():
    for d in [
        "27_confirmation_derivatives_c1b/encoded",
        "27_confirmation_derivatives_c1b/decoded_raw",
        "27_confirmation_derivatives_c1b/aligned_pcm",
        "27_confirmation_derivatives_c1b/audit",
        "27_confirmation_derivatives_c1b/scratch",
        "28_confirmation_features_c1b/representation_a",
        "28_confirmation_features_c1b/clean_reference",
        "28_confirmation_features_c1b/audit",
        "29_reference_c1b/centroids",
        "29_reference_c1b/operational_lock",
        "29_reference_c1b/integrity",
        "30_results_c1b/predictions",
        "30_results_c1b/per_rate",
        "30_results_c1b/bootstrap",
        "30_results_c1b/corpus",
        "30_results_c1b/geometry",
        "30_results_c1b/gate",
        "31_reports_c1b",
        "logs_c1b",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    fixed_paths = {
        "AQ2C1A_FREEZE": ROOT / "03_manifests" / "AQ2C1A_FREEZE.json",
        "PROSPECTIVE_SPEC": ROOT / "25_design_c1a" / "AQ2C1B_PROSPECTIVE_CONFIRMATION_SPEC.json",
        "CONFIRM_PARENT_MANIFEST": ROOT / "23_confirmation_corpus_c1a" / "AQ2C1A_confirmation_parent_manifest.csv",
        "CONFIRM_PCM_HASHES": ROOT / "23_confirmation_corpus_c1a" / "AQ2C1A_confirmation_pcm_sha256s.txt",
        "DISCOVERY_EXCLUSION": ROOT / "23_confirmation_corpus_c1a" / "selection_audit" / "AQ2C1A_discovery_parent_exclusion.csv",
        "MODEL_NUMERIC": ROOT / "18_model_b1" / "AQ2B1_model_numeric.npz",
        "MODEL_CONFIG": ROOT / "18_model_b1" / "AQ2B1_model_config.json",
        "PRETEST": ROOT / "18_model_b1" / "AQ2B1_PRETEST_MODEL_FREEZE.json",
        "FEATURE_SPEC": ROOT / "14_feature_preflight_b0" / "AQ2B0_feature_spec.json",
        "FEATURE_EXTRACTOR": ROOT / "14_feature_preflight_b0" / "aq2b0_feature_extractor.py",
        "PYTHON": PYTHON,
        "B1_DERIVATIVE_MANIFEST": ROOT / "16_derivatives_b1" / "AQ2B1_derivative_manifest.csv",
        "B1_FEATURE_MANIFEST": ROOT / "17_features_b1" / "AQ2B1_feature_manifest.csv",
    }
    for key, path in fixed_paths.items():
        if key in EXPECTED and sha256_file(path) != EXPECTED[key]:
            fail(f"{key}_IDENTITY_MISMATCH", {"actual_sha256": sha256_file(path)})

    c1a = json.loads(fixed_paths["AQ2C1A_FREEZE"].read_text(encoding="utf-8"))
    req_c1a = {
        "final_classification": "PASS_CONFIRMATION_DESIGN_FROZEN",
        "confirmation_parent_count": 60,
        "confirmation_parents_per_corpus": 20,
        "confirmation_parents_untouched": "YES",
        "discovery_parent_overlap_count": 0,
        "discovery_pcm_collision_count": 0,
        "within_confirmation_pcm_duplicate_count": 0,
        "preflight_attempt_count": 54,
        "preflight_encode_success_count": 54,
        "preflight_probe_success_count": 54,
        "preflight_decode_success_count": 54,
        "preflight_alignment_success_count": 54,
        "expected_C1B_derivative_count": 1080,
        "confirmation_prediction_count": 0,
        "confirmation_model_predictions_generated": "NO",
        "scientific_confirmation_result_produced": "NO",
        "new_training_performed": "NO",
        "model_refit_performed": "NO",
        "new_model_weights_created": "NO",
        "vorbis_used": "NO",
        "representation_B_used": "NO",
        "C1A_PREFLIGHT_DERIVATIVES_SCIENTIFIC_USE": "NO",
    }
    for key, expected in req_c1a.items():
        if c1a.get(key) != expected:
            fail("AQ2C1A_IDENTITY_MISMATCH", {"key": key, "expected": expected, "actual": c1a.get(key)})

    spec = json.loads(fixed_paths["PROSPECTIVE_SPEC"].read_text(encoding="utf-8"))
    if spec.get("primary_confirmation_family") != "AAC" or spec.get("codec_set") != CLASS_ORDER or spec.get("confirmation_parent_count") != 60 or spec.get("parents_per_corpus") != 20 or spec.get("confirmation_parent_manifest_sha256") != EXPECTED["CONFIRM_PARENT_MANIFEST"] or spec.get("rate_grid") != RATES or spec.get("new_probe_rates") != NEW_RATES or spec.get("discovery_replication_rate") != 64 or spec.get("seen_endpoint_rates") != [32, 128] or spec.get("expected_C1B_derivative_count") != 1080 or spec.get("CONFIRMATION PREDICTIONS OBSERVED BEFORE SPEC FREEZE") != "NO":
        fail("PROSPECTIVE_SPEC_IDENTITY_MISMATCH", spec)

    parents = pd.read_csv(fixed_paths["CONFIRM_PARENT_MANIFEST"])
    if len(parents) != 60 or parents["confirmation_parent_uid"].nunique() != 60 or parents["source_corpus"].value_counts().to_dict() != {"FSDnoisy18k": 20, "IRMAS": 20, "Orchset": 20}:
        fail("CONFIRMATION_PARENT_COHORT_MISMATCH", {"count": len(parents), "counts": parents["source_corpus"].value_counts().to_dict()})
    for rec in parents.to_dict("records"):
        p = Path(rec["canonical_pcm_path"])
        arr, info = wav_read(p)
        if info != {"sample_rate": 48000, "channels": 1, "sample_width": 2, "sample_count": 96000} or raw_pcm_sha256(p) != rec["pcm_sha256"]:
            fail("CONFIRMATION_PARENT_PCM_MISMATCH", {"parent": rec["confirmation_parent_uid"]})

    exclusion = pd.read_csv(fixed_paths["DISCOVERY_EXCLUSION"])
    disc_uids = set(exclusion["parent_uid"])
    disc_pcm = set(exclusion["canonical_pcm_sha256"])
    overlap = len(set(parents["upstream_parent_uid"]) & disc_uids)
    pcm_collision = len(set(parents["pcm_sha256"]) & disc_pcm)
    within_pcm_dup = len(parents) - parents["pcm_sha256"].nunique()
    if len(exclusion) != 82 or overlap or pcm_collision or within_pcm_dup:
        fail("CONFIRMATION_PARENT_CONTAMINATION", {"discovery_rows": len(exclusion), "overlap": overlap, "pcm_collision": pcm_collision, "within_dup": within_pcm_dup})

    r2 = json.loads((ROOT / "03_manifests" / "AQ2A_R2_FREEZE.json").read_text(encoding="utf-8"))
    encoder = Path(r2["r2_encoder_ffmpeg_path"])
    ffprobe = Path(r2["r2_ffprobe_path"])
    decoder = Path(r2["common_decoder_path"])
    if sha256_file(encoder) != EXPECTED["ENCODER"] or sha256_file(ffprobe) != EXPECTED["FFPROBE"] or sha256_file(decoder) != EXPECTED["DECODER"]:
        fail("TOOLCHAIN_IDENTITY_MISMATCH", {})
    if platform.python_version() != "3.12.3" or np.__version__ != "1.26.4" or scipy.__version__ != "1.13.1" or pd.__version__ != "2.2.2" or sklearn.__version__ != "1.5.1":
        fail("PYTHON_ENVIRONMENT_MISMATCH", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__})

    model = np.load(fixed_paths["MODEL_NUMERIC"], allow_pickle=False)
    mean = model["scaler_mean"]
    scale = model["scaler_scale"]
    coef = model["model_coef"]
    intercept = model["model_intercept"]
    if list(model["class_order"]) != CLASS_ORDER or list(model["sklearn_classes"]) != CLASS_ORDER:
        fail("MODEL_CLASS_ORDER_MISMATCH", {"class_order": list(model["class_order"])})

    b1_deriv = pd.read_csv(fixed_paths["B1_DERIVATIVE_MANIFEST"])
    b1_features_manifest = pd.read_csv(fixed_paths["B1_FEATURE_MANIFEST"])
    b1_path_by_uid = dict(zip(b1_features_manifest["derivative_uid"], b1_features_manifest["npy_path"]))
    train = b1_deriv[b1_deriv["split"] == "TRAIN"].copy()
    if len(train) != 294 or train["codec_family"].value_counts().to_dict() != {"AAC": 98, "MP3": 98, "Opus": 98}:
        fail("B1_TRAIN_CENTROID_INPUT_MISMATCH", {"count": len(train), "counts": train["codec_family"].value_counts().to_dict()})
    centroids = {}
    for codec in CLASS_ORDER:
        zs = []
        for uid in train.loc[train["codec_family"] == codec, "derivative_uid"]:
            x = np.load(b1_path_by_uid[uid], allow_pickle=False)
            zs.append((x - mean) / scale)
        centroids[codec] = np.mean(np.vstack(zs), axis=0)
    centroid_path = ROOT / "29_reference_c1b" / "centroids" / "AQ2C1B_B1_train_centroids.npz"
    np.savez(centroid_path, centroid_AAC=centroids["AAC"], centroid_MP3=centroids["MP3"], centroid_Opus=centroids["Opus"], class_order=np.asarray(CLASS_ORDER))
    operational_spec = {
        "centroid_definition": "Mean standardized B1 TRAIN Representation-A vectors per true codec.",
        "standardization": "Frozen B1 scaler_mean and scaler_scale only; no refit.",
        "distance_metric": "Euclidean L2",
        "wrong_centroid_closer_rule": "D_WRONG_MIN < D_TRUE, strict inequality; ties are not wrong.",
        "true_centroid_rank_rule": "1 + number of wrong-codec centroids with distance strictly less than D_TRUE",
        "class_order": CLASS_ORDER,
    }
    operational_spec_path = ROOT / "29_reference_c1b" / "operational_lock" / "AQ2C1B_OPERATIONAL_METRIC_SPEC.json"
    write_json(operational_spec_path, operational_spec)

    def centroid_eval(z, true_codec):
        d = {c: float(np.linalg.norm(z - centroids[c])) for c in CLASS_ORDER}
        d_true = d[true_codec]
        rank = 1 + sum(1 for c in CLASS_ORDER if c != true_codec and d[c] < d_true)
        wrong = min(d[c] for c in CLASS_ORDER if c != true_codec) < d_true
        return d, rank, wrong

    mid_test = b1_deriv[(b1_deriv["split"] == "TEST") & (b1_deriv["nominal_rate_kbps"] == 64)]
    centroid_audit = {"status": "PASS", "observed": {}}
    expected_centroid = {
        "AAC": {"mid_fraction_rank_1": 0.0, "mid_fraction_wrong_closer": 1.0},
        "MP3": {"mid_fraction_rank_1": 0.35294117647058826, "mid_fraction_wrong_closer": 0.6470588235294118},
        "Opus": {"mid_fraction_rank_1": 0.7647058823529411, "mid_fraction_wrong_closer": 0.23529411764705882},
    }
    for codec in CLASS_ORDER:
        ranks = []
        wrongs = []
        for uid in mid_test.loc[mid_test["codec_family"] == codec, "derivative_uid"]:
            x = np.load(b1_path_by_uid[uid], allow_pickle=False)
            _, rank, wrong = centroid_eval((x - mean) / scale, codec)
            ranks.append(rank == 1)
            wrongs.append(wrong)
        obs = {"mid_fraction_rank_1": float(np.mean(ranks)), "mid_fraction_wrong_closer": float(np.mean(wrongs))}
        centroid_audit["observed"][codec] = obs
        if any(abs(obs[k] - expected_centroid[codec][k]) > 1e-15 for k in obs):
            fail("CENTROID_OPERATIONALIZATION_MISMATCH", {"codec": codec, "observed": obs, "expected": expected_centroid[codec]})
    centroid_audit_path = ROOT / "29_reference_c1b" / "integrity" / "AQ2C1B_centroid_reproduction_audit.json"
    write_json(centroid_audit_path, centroid_audit)

    model_audit_rows = []
    for pred_path in [ROOT / "19_results_b1" / "predictions" / "AQ2B1_test_seen_predictions.csv", ROOT / "19_results_b1" / "predictions" / "AQ2B1_test_mid_predictions.csv"]:
        pred = pd.read_csv(pred_path)
        for rec in pred.to_dict("records"):
            x = np.load(b1_path_by_uid[rec["derivative_uid"]], allow_pickle=False)
            z = ((x - mean) / scale)[None, :]
            probs = stable_softmax(z @ coef.T + intercept)[0]
            reproduced = CLASS_ORDER[int(np.argmax(probs))]
            model_audit_rows.append({"prediction_file": str(pred_path), "derivative_uid": rec["derivative_uid"], "true_codec": rec["true_codec"], "frozen_predicted_codec": rec["predicted_codec"], "reproduced_predicted_codec": reproduced, "prediction_match": reproduced == rec["predicted_codec"]})
    model_audit_path = ROOT / "29_reference_c1b" / "integrity" / "AQ2C1B_model_numeric_reproduction_audit.csv"
    write_csv(model_audit_path, model_audit_rows, list(model_audit_rows[0].keys()))
    if not all(r["prediction_match"] for r in model_audit_rows):
        fail("FROZEN_MODEL_NUMERIC_REPRODUCTION_FAIL", {"mismatches": [r for r in model_audit_rows if not r["prediction_match"]][:10]})

    preconfirm_lock_path = ROOT / "29_reference_c1b" / "operational_lock" / "AQ2C1B_PRECONFIRM_OPERATIONAL_LOCK.json"
    preconfirm_lock = {
        "AQ2C1A_freeze_sha256": EXPECTED["AQ2C1A_FREEZE"],
        "prospective_spec_sha256": EXPECTED["PROSPECTIVE_SPEC"],
        "confirmation_parent_manifest_sha256": EXPECTED["CONFIRM_PARENT_MANIFEST"],
        "model_numeric_sha256": EXPECTED["MODEL_NUMERIC"],
        "model_config_sha256": EXPECTED["MODEL_CONFIG"],
        "feature_spec_sha256": EXPECTED["FEATURE_SPEC"],
        "feature_extractor_sha256": EXPECTED["FEATURE_EXTRACTOR"],
        "centroid_numeric_sha256": sha256_file(centroid_path),
        "operational_metric_spec_sha256": sha256_file(operational_spec_path),
        "centroid_historical_reproduction": "PASS",
        "model_historical_reproduction": "PASS",
        "confirmation_prediction_count": 0,
    }
    write_json(preconfirm_lock_path, preconfirm_lock)

    derivative_rows = []
    source_samples = {}
    for rec in parents.sort_values(["source_corpus", "confirmation_parent_uid"]).to_dict("records"):
        parent_uid = rec["confirmation_parent_uid"]
        src_path = Path(rec["canonical_pcm_path"])
        if parent_uid not in source_samples:
            source_samples[parent_uid] = wav_read(src_path)[0]
        for codec in CLASS_ORDER:
            for rate in RATES:
                ci = codec_info(codec, rate)
                uid_source = f"AQ2C1B_DERIV_V1|{parent_uid}|{codec}|{rate}|{ci['core']}|{EXPECTED['ENCODER']}"
                duid = sha256_text(uid_source)[:24]
                encoded = ROOT / "27_confirmation_derivatives_c1b" / "encoded" / f"{duid}.{ci['ext']}"
                decoded = ROOT / "27_confirmation_derivatives_c1b" / "decoded_raw" / f"{duid}.wav"
                aligned = ROOT / "27_confirmation_derivatives_c1b" / "aligned_pcm" / f"{duid}.wav"
                enc_cmd = [str(encoder), "-y", "-hide_banner", "-i", str(src_path), "-map_metadata", "-1", "-vn", "-sn", "-dn"] + ci["args"] + [str(encoded)]
                enc = run_cmd(enc_cmd, timeout=120)
                enc_ok = enc["return_code"] == 0 and encoded.exists()
                probe_cmd = [str(ffprobe), "-v", "error", "-show_entries", "stream=codec_name,codec_long_name,profile,sample_rate,channels,duration,bit_rate:format=format_name,format_long_name,duration,bit_rate", "-of", "json", str(encoded)]
                probe = run_cmd(probe_cmd, timeout=60) if enc_ok else {"command": " ".join(probe_cmd), "return_code": 1, "stdout": "", "stderr": "encode failed"}
                probe_fields = {}
                probe_ok = False
                try:
                    js = json.loads(probe["stdout"]) if probe["return_code"] == 0 else {}
                except json.JSONDecodeError:
                    js = {}
                stream = js.get("streams", [{}])[0] if js.get("streams") else {}
                fmt = js.get("format", {})
                probe_fields = {
                    "probe_codec_name": stream.get("codec_name", ""),
                    "probe_codec_long_name": stream.get("codec_long_name", ""),
                    "probe_profile": stream.get("profile", ""),
                    "probe_sample_rate": stream.get("sample_rate", ""),
                    "probe_channels": stream.get("channels", ""),
                    "probe_duration": stream.get("duration", fmt.get("duration", "")),
                    "probe_bit_rate": stream.get("bit_rate", fmt.get("bit_rate", "")),
                    "probe_format_name": fmt.get("format_name", ""),
                    "probe_format_long_name": fmt.get("format_long_name", ""),
                }
                probe_ok = probe["return_code"] == 0 and probe_fields["probe_codec_name"] == ci["probe"]
                dec_cmd = [str(decoder), "-y", "-hide_banner", "-i", str(encoded), "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(decoded)]
                dec = run_cmd(dec_cmd, timeout=120) if probe_ok else {"command": " ".join(dec_cmd), "return_code": 1, "stdout": "", "stderr": "probe failed"}
                dec_ok = dec["return_code"] == 0 and decoded.exists()
                align = {"lag": "", "corr": "", "post_count": "", "leading_crop": "", "success": False}
                aligned_sha = ""
                if dec_ok:
                    dec_samples, dec_info = wav_read(decoded)
                    if dec_info["sample_rate"] == 48000 and dec_info["channels"] == 1 and dec_info["sample_width"] == 2:
                        align = best_alignment(source_samples[parent_uid], dec_samples)
                        if align["success"] and align["post_count"] == SEGMENT_SAMPLES:
                            start = int(align["leading_crop"])
                            wav_write(aligned, dec_samples[start : start + SEGMENT_SAMPLES])
                            aligned_sha = raw_pcm_sha256(aligned)
                aligned_ok = aligned.exists() and align["success"] and align["post_count"] == SEGMENT_SAMPLES
                row = {
                    "derivative_uid": duid,
                    "confirmation_parent_uid": parent_uid,
                    "source_corpus": rec["source_corpus"],
                    "codec_family": codec,
                    "encoder_name": ci["encoder"],
                    "encoder_core": ci["core"],
                    "nominal_rate_kbps": rate,
                    "achieved_rate_kbps": (8 * encoded.stat().st_size / SEGMENT_SECONDS / 1000.0) if encoded.exists() else "",
                    "encoded_sha256": sha256_file(encoded) if encoded.exists() else "",
                    "aligned_pcm_sha256": aligned_sha,
                    "alignment_lag_samples": align["lag"],
                    "aligned_sample_count": SEGMENT_SAMPLES if aligned_ok else "",
                    "encoded_size_bytes": encoded.stat().st_size if encoded.exists() else "",
                    "encoded_path": str(encoded) if encoded.exists() else "",
                    "aligned_pcm_path": str(aligned) if aligned.exists() else "",
                    "derivative_id_source_string": uid_source,
                    "encode_return_code": enc["return_code"],
                    "probe_return_code": probe["return_code"],
                    "decode_return_code": dec["return_code"],
                    "status": "PASS" if enc_ok and probe_ok and dec_ok and aligned_ok else "FAIL",
                    **probe_fields,
                    "encode_command": enc["command"],
                    "probe_command": probe["command"],
                    "decode_command": dec["command"],
                }
                derivative_rows.append(row)
    deriv_fields = ["derivative_uid", "confirmation_parent_uid", "source_corpus", "codec_family", "encoder_name", "encoder_core", "nominal_rate_kbps", "achieved_rate_kbps", "encoded_sha256", "aligned_pcm_sha256", "alignment_lag_samples", "aligned_sample_count", "encoded_size_bytes", "encoded_path", "aligned_pcm_path", "derivative_id_source_string", "encode_return_code", "probe_return_code", "decode_return_code", "status", "probe_codec_name", "probe_codec_long_name", "probe_profile", "probe_sample_rate", "probe_channels", "probe_duration", "probe_bit_rate", "probe_format_name", "probe_format_long_name", "encode_command", "probe_command", "decode_command"]
    derivative_manifest_path = ROOT / "27_confirmation_derivatives_c1b" / "AQ2C1B_derivative_manifest.csv"
    write_csv(derivative_manifest_path, derivative_rows, deriv_fields)
    if len(derivative_rows) != 1080 or not all(r["status"] == "PASS" for r in derivative_rows) or len({r["derivative_uid"] for r in derivative_rows}) != 1080:
        fail("CONFIRMATION_MATRIX_INCOMPLETE", {"rows": len(derivative_rows), "pass": sum(1 for r in derivative_rows if r["status"] == "PASS"), "unique_uids": len({r["derivative_uid"] for r in derivative_rows})})

    with open(ROOT / "27_confirmation_derivatives_c1b" / "AQ2C1B_derivative_file_sha256s.txt", "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(list((ROOT / "27_confirmation_derivatives_c1b" / "encoded").glob("*")) + list((ROOT / "27_confirmation_derivatives_c1b" / "decoded_raw").glob("*")) + list((ROOT / "27_confirmation_derivatives_c1b" / "aligned_pcm").glob("*")), key=lambda x: str(x.relative_to(ROOT)).replace("\\", "/")):
            if p.is_file():
                f.write(f"{sha256_file(p)}  {str(p.relative_to(ROOT)).replace('\\', '/')}\n")

    collision_rows = []
    by_pcm = defaultdict(list)
    for r in derivative_rows:
        by_pcm[r["aligned_pcm_sha256"]].append(r)
    for pcm, rows in by_pcm.items():
        if len(rows) > 1:
            labels = sorted({r["codec_family"] for r in rows})
            rates = sorted({int(r["nominal_rate_kbps"]) for r in rows})
            collision_rows.append({"aligned_pcm_sha256": pcm, "count": len(rows), "codec_labels": ";".join(labels), "rates": ";".join(map(str, rates)), "cross_codec": len(labels) > 1, "cross_rate": len(rates) > 1, "derivative_uids": ";".join(r["derivative_uid"] for r in rows)})
    if not collision_rows:
        collision_rows = [{"aligned_pcm_sha256": "", "count": 0, "codec_labels": "", "rates": "", "cross_codec": False, "cross_rate": False, "derivative_uids": ""}]
    collision_path = ROOT / "27_confirmation_derivatives_c1b" / "audit" / "AQ2C1B_collision_audit.csv"
    write_csv(collision_path, collision_rows, list(collision_rows[0].keys()))

    achieved_summary_rows = []
    for codec in CLASS_ORDER:
        for rate in RATES:
            vals = [float(r["achieved_rate_kbps"]) for r in derivative_rows if r["codec_family"] == codec and int(r["nominal_rate_kbps"]) == rate]
            s = summary_stats(vals)
            achieved_summary_rows.append({"codec": codec, "nominal_rate_kbps": rate, **s, "n": len(vals)})
    achieved_summary_path = ROOT / "27_confirmation_derivatives_c1b" / "audit" / "AQ2C1B_achieved_rate_summary.csv"
    write_csv(achieved_summary_path, achieved_summary_rows, list(achieved_summary_rows[0].keys()))

    extractor = load_extractor()
    feature_spec = json.loads(fixed_paths["FEATURE_SPEC"].read_text(encoding="utf-8"))
    feature_rows = []
    feature_by_uid = {}
    for r in derivative_rows:
        x = extractor.read_pcm16_wav(r["aligned_pcm_path"])
        G = extractor.extract_stft_core(x, feature_spec)
        arr = extractor.extract_representation_a(G, feature_spec)
        if arr.shape != (256,) or arr.dtype != np.float64 or not np.all(np.isfinite(arr)):
            fail("FEATURE_INTEGRITY_FAIL", {"derivative_uid": r["derivative_uid"]})
        out = ROOT / "28_confirmation_features_c1b" / "representation_a" / f"{r['derivative_uid']}.npy"
        np.save(out, arr, allow_pickle=False)
        feature_by_uid[r["derivative_uid"]] = arr
        feature_rows.append({"derivative_uid": r["derivative_uid"], "array_sha256": array_hash(arr), "npy_file_sha256": sha256_file(out), "shape": "256", "dtype": str(arr.dtype), "finite_check": True, "npy_path": str(out)})
    feature_manifest_path = ROOT / "28_confirmation_features_c1b" / "AQ2C1B_feature_manifest.csv"
    write_csv(feature_manifest_path, feature_rows, list(feature_rows[0].keys()))

    clean_rows = []
    clean_by_parent = {}
    for rec in parents.to_dict("records"):
        x = extractor.read_pcm16_wav(rec["canonical_pcm_path"])
        G = extractor.extract_stft_core(x, feature_spec)
        arr = extractor.extract_representation_a(G, feature_spec)
        out = ROOT / "28_confirmation_features_c1b" / "clean_reference" / f"{rec['confirmation_parent_uid']}.npy"
        np.save(out, arr, allow_pickle=False)
        clean_by_parent[rec["confirmation_parent_uid"]] = arr
        clean_rows.append({"confirmation_parent_uid": rec["confirmation_parent_uid"], "source_corpus": rec["source_corpus"], "array_sha256": array_hash(arr), "npy_file_sha256": sha256_file(out), "shape": "256", "dtype": str(arr.dtype), "finite_check": True, "npy_path": str(out)})
    clean_feature_manifest_path = ROOT / "28_confirmation_features_c1b" / "AQ2C1B_clean_feature_manifest.csv"
    write_csv(clean_feature_manifest_path, clean_rows, list(clean_rows[0].keys()))
    with open(ROOT / "28_confirmation_features_c1b" / "AQ2C1B_feature_file_sha256s.txt", "w", encoding="utf-8", newline="\n") as f:
        for p in sorted(list((ROOT / "28_confirmation_features_c1b" / "representation_a").glob("*.npy")) + list((ROOT / "28_confirmation_features_c1b" / "clean_reference").glob("*.npy")), key=lambda x: str(x.relative_to(ROOT)).replace("\\", "/")):
            f.write(f"{sha256_file(p)}  {str(p.relative_to(ROOT)).replace('\\', '/')}\n")

    preprediction_path = ROOT / "03_manifests" / "AQ2C1B_PREPREDICTION_FREEZE.json"
    preprediction = {
        "phase": "AQ-2C.1B",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "AQ2C1A_freeze_sha256": EXPECTED["AQ2C1A_FREEZE"],
        "prospective_confirmation_spec_sha256": EXPECTED["PROSPECTIVE_SPEC"],
        "preconfirm_operational_lock_sha256": sha256_file(preconfirm_lock_path),
        "confirmation_parent_manifest_sha256": EXPECTED["CONFIRM_PARENT_MANIFEST"],
        "confirmation_parent_count": 60,
        "derivative_expected": 1080,
        "derivative_actual": len(derivative_rows),
        "encode_success_count": 1080,
        "probe_success_count": 1080,
        "decode_success_count": 1080,
        "alignment_success_count": 1080,
        "feature_success_count": len(feature_rows),
        "clean_feature_success_count": len(clean_rows),
        "derivative_manifest_sha256": sha256_file(derivative_manifest_path),
        "feature_manifest_sha256": sha256_file(feature_manifest_path),
        "clean_feature_manifest_sha256": sha256_file(clean_feature_manifest_path),
        "collision_audit_sha256": sha256_file(collision_path),
        "derivative_checksum_manifest_sha256": sha256_file(ROOT / "27_confirmation_derivatives_c1b" / "AQ2C1B_derivative_file_sha256s.txt"),
        "feature_checksum_manifest_sha256": sha256_file(ROOT / "28_confirmation_features_c1b" / "AQ2C1B_feature_file_sha256s.txt"),
        "model_numeric_sha256": EXPECTED["MODEL_NUMERIC"],
        "model_config_sha256": EXPECTED["MODEL_CONFIG"],
        "centroid_numeric_sha256": sha256_file(centroid_path),
        "feature_spec_sha256": EXPECTED["FEATURE_SPEC"],
        "feature_extractor_sha256": EXPECTED["FEATURE_EXTRACTOR"],
        "confirmation_prediction_count": 0,
        "scientific_confirmation_metrics_computed": "NO",
    }
    write_json(preprediction_path, preprediction)
    preprediction_sha = sha256_file(preprediction_path)

    pred_rows = []
    z_by_uid = {}
    for r in derivative_rows:
        x = feature_by_uid[r["derivative_uid"]]
        z = (x - mean) / scale
        z_by_uid[r["derivative_uid"]] = z
        logits = (z[None, :] @ coef.T + intercept)[0]
        probs = stable_softmax(logits[None, :])[0]
        true_codec = r["codec_family"]
        true_idx = CLASS_ORDER.index(true_codec)
        wrong_logits = [float(logits[i]) for i in range(3) if i != true_idx]
        dists, rank, wrong_centroid = centroid_eval(z, true_codec)
        pred_rows.append({
            "derivative_uid": r["derivative_uid"],
            "confirmation_parent_uid": r["confirmation_parent_uid"],
            "source_corpus": r["source_corpus"],
            "true_codec": true_codec,
            "nominal_rate_kbps": int(r["nominal_rate_kbps"]),
            "predicted_codec": CLASS_ORDER[int(np.argmax(logits))],
            "prob_AAC": float(probs[0]),
            "prob_MP3": float(probs[1]),
            "prob_Opus": float(probs[2]),
            "true_margin": float(logits[true_idx] - max(wrong_logits)),
            "distance_centroid_AAC": dists["AAC"],
            "distance_centroid_MP3": dists["MP3"],
            "distance_centroid_Opus": dists["Opus"],
            "true_centroid_rank": rank,
            "wrong_centroid_closer": bool(wrong_centroid),
            "achieved_rate_kbps": float(r["achieved_rate_kbps"]),
        })
    pred_path = ROOT / "30_results_c1b" / "predictions" / "AQ2C1B_all_predictions.csv"
    write_csv(pred_path, pred_rows, list(pred_rows[0].keys()))

    per_rate = {str(rate): metrics_for([r for r in pred_rows if r["nominal_rate_kbps"] == rate]) for rate in RATES}
    per_rate_path = ROOT / "30_results_c1b" / "per_rate" / "AQ2C1B_per_rate_metrics.json"
    write_json(per_rate_path, per_rate)
    codec_rate_rows = []
    for codec in CLASS_ORDER:
        for rate in RATES:
            items = [r for r in pred_rows if r["true_codec"] == codec and r["nominal_rate_kbps"] == rate]
            recall = sum(1 for r in items if r["predicted_codec"] == codec) / len(items)
            target, share = dominant_error(items, codec)
            codec_rate_rows.append({
                "codec": codec,
                "nominal_rate_kbps": rate,
                "n": len(items),
                "recall": recall,
                "median_TRUE_MARGIN": float(np.median([r["true_margin"] for r in items])),
                "fraction_TRUE_MARGIN_gt_0": float(np.mean([r["true_margin"] > 0 for r in items])),
                "fraction_true_centroid_rank_1": float(np.mean([r["true_centroid_rank"] == 1 for r in items])),
                "fraction_wrong_centroid_closer": float(np.mean([r["wrong_centroid_closer"] for r in items])),
                "dominant_error_target": target,
                "dominant_error_share": share,
            })
    per_codec_rate_path = ROOT / "30_results_c1b" / "per_rate" / "AQ2C1B_per_codec_rate_summary.csv"
    write_csv(per_codec_rate_path, codec_rate_rows, list(codec_rate_rows[0].keys()))

    endpoint_rows = [r for r in pred_rows if r["nominal_rate_kbps"] in [32, 128]]
    endpoint_metrics = metrics_for(endpoint_rows)
    aac32 = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == 32)["recall"]
    aac128 = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == 128)["recall"]
    endpoint_floor = min(aac32, aac128)
    endpoint_pass = endpoint_metrics["balanced_accuracy"] >= 0.75 and aac32 >= 0.60 and aac128 >= 0.60
    domain_severe = endpoint_metrics["balanced_accuracy"] < 0.65 or (aac32 < 0.50 and aac128 < 0.50)

    parent_aac = {}
    for parent in parents["confirmation_parent_uid"]:
        parent_aac[parent] = {}
        for rate in RATES:
            row = next(r for r in pred_rows if r["confirmation_parent_uid"] == parent and r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == rate)
            parent_aac[parent][rate] = row["predicted_codec"] == "AAC"
    bootstrap_summary = {}
    for rate in BOOT_RATES:
        seed = 20260901 + rate
        rng = np.random.default_rng(seed)
        parent_ids = list(parent_aac)
        drops = np.asarray([0.5 * (parent_aac[p][32] + parent_aac[p][128]) - parent_aac[p][rate] for p in parent_ids], dtype=np.float64)
        reps = []
        for i in range(10000):
            idx = rng.integers(0, len(parent_ids), len(parent_ids))
            reps.append({"replicate": i, "mean_drop": float(np.mean(drops[idx]))})
        path = ROOT / "30_results_c1b" / "bootstrap" / f"AQ2C1B_AAC_parent_bootstrap_{rate}.csv"
        write_csv(path, reps, ["replicate", "mean_drop"])
        vals = np.asarray([r["mean_drop"] for r in reps])
        bootstrap_summary[str(rate)] = {"seed": seed, "replicates": 10000, "p2_5": float(np.percentile(vals, 2.5)), "p50": float(np.percentile(vals, 50)), "p97_5": float(np.percentile(vals, 97.5)), "PAIRED_DROP_STABLE": bool(np.percentile(vals, 2.5) > 0)}
    bootstrap_summary_path = ROOT / "30_results_c1b" / "bootstrap" / "AQ2C1B_AAC_parent_bootstrap_summary.json"
    write_json(bootstrap_summary_path, bootstrap_summary)

    corpus_rows = []
    for corpus in CORPORA:
        for rate in RATES:
            items = [r for r in pred_rows if r["source_corpus"] == corpus and r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == rate]
            corpus_rows.append({"source_corpus": corpus, "nominal_rate_kbps": rate, "AAC_recall": float(np.mean([r["predicted_codec"] == "AAC" for r in items])), "n": len(items)})
    corpus_path = ROOT / "30_results_c1b" / "corpus" / "AQ2C1B_AAC_corpus_rate_summary.csv"
    write_csv(corpus_path, corpus_rows, list(corpus_rows[0].keys()))
    corpus_lookup = {(r["source_corpus"], r["nominal_rate_kbps"]): r["AAC_recall"] for r in corpus_rows}
    multi_corpus = {}
    for rate in NEW_RATES:
        details = {}
        count = 0
        for corpus in CORPORA:
            degrade = corpus_lookup[(corpus, rate)] < 0.5 * (corpus_lookup[(corpus, 32)] + corpus_lookup[(corpus, 128)])
            details[corpus] = bool(degrade)
            count += int(degrade)
        multi_corpus[str(rate)] = {"corpus_degradation": details, "true_count": count, "MULTI_CORPUS_CONFIRMATION": bool(count >= 2)}
    multi_corpus_path = ROOT / "30_results_c1b" / "corpus" / "AQ2C1B_multi_corpus_gate.json"
    write_json(multi_corpus_path, multi_corpus)

    clean_distance_rows = []
    for r in derivative_rows:
        uid = r["derivative_uid"]
        parent = r["confirmation_parent_uid"]
        x_lossy = feature_by_uid[uid]
        x_clean = clean_by_parent[parent]
        clean_distance_rows.append({
            "derivative_uid": uid,
            "confirmation_parent_uid": parent,
            "source_corpus": r["source_corpus"],
            "codec": r["codec_family"],
            "nominal_rate_kbps": int(r["nominal_rate_kbps"]),
            "RAW_D_TO_CLEAN": float(np.linalg.norm(x_lossy - x_clean)),
            "STD_D_TO_CLEAN": float(np.linalg.norm((x_lossy - mean) / scale - (x_clean - mean) / scale)),
        })
    clean_distance_path = ROOT / "30_results_c1b" / "geometry" / "AQ2C1B_clean_distance.csv"
    write_csv(clean_distance_path, clean_distance_rows, list(clean_distance_rows[0].keys()))

    uid_by_parent_codec_rate = {(r["confirmation_parent_uid"], r["codec_family"], int(r["nominal_rate_kbps"])): r["derivative_uid"] for r in derivative_rows}
    trajectory_rows = []
    for parent in parents["confirmation_parent_uid"]:
        corpus = str(parents.loc[parents["confirmation_parent_uid"] == parent, "source_corpus"].iloc[0])
        for codec in CLASS_ORDER:
            z32 = z_by_uid[uid_by_parent_codec_rate[(parent, codec, 32)]]
            z128 = z_by_uid[uid_by_parent_codec_rate[(parent, codec, 128)]]
            v = z128 - z32
            denom = float(np.dot(v, v))
            endpoint = float(np.linalg.norm(v))
            for rate in [48, 64, 80, 96]:
                z = z_by_uid[uid_by_parent_codec_rate[(parent, codec, rate)]]
                if denom == 0:
                    alpha = None
                    resid = None
                    norm = None
                    within = None
                else:
                    alpha = float(np.dot(z - z32, v) / denom)
                    proj = z32 + alpha * v
                    resid = float(np.linalg.norm(z - proj))
                    norm = resid / endpoint
                    within = bool(0 <= alpha <= 1)
                trajectory_rows.append({"confirmation_parent_uid": parent, "source_corpus": corpus, "codec": codec, "nominal_rate_kbps": rate, "ALPHA": alpha, "ORTHOGONAL_RESIDUAL": resid, "ENDPOINT_DISTANCE": endpoint, "NORMALIZED_RESIDUAL": norm, "WITHIN_ENDPOINT_SEGMENT": within})
    trajectory_path = ROOT / "30_results_c1b" / "geometry" / "AQ2C1B_endpoint_trajectory_geometry.csv"
    write_csv(trajectory_path, trajectory_rows, list(trajectory_rows[0].keys()))

    rate_events = {}
    for rate in NEW_RATES:
        cr = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == rate)
        drop = endpoint_floor - cr["recall"]
        structural = cr["recall"] <= 0.50 and drop >= 0.25 and cr["median_TRUE_MARGIN"] < 0 and cr["fraction_wrong_centroid_closer"] >= 0.60 and cr["dominant_error_share"] is not None and cr["dominant_error_share"] >= 0.60
        rate_events[str(rate)] = {
            "AAC_RECALL": cr["recall"],
            "AAC_RECALL_DROP": drop,
            "AAC_MEDIAN_TRUE_MARGIN": cr["median_TRUE_MARGIN"],
            "AAC_WRONG_CENTROID_FRACTION": cr["fraction_wrong_centroid_closer"],
            "AAC_DOMINANT_ERROR_TARGET": cr["dominant_error_target"],
            "AAC_DOMINANT_ERROR_SHARE": cr["dominant_error_share"],
            "NEW_RATE_STRUCTURAL_EVENT": bool(structural),
            "PAIRED_DROP_BOOTSTRAP_LOW": bootstrap_summary[str(rate)]["p2_5"],
            "PAIRED_DROP_BOOTSTRAP_MEDIAN": bootstrap_summary[str(rate)]["p50"],
            "PAIRED_DROP_BOOTSTRAP_HIGH": bootstrap_summary[str(rate)]["p97_5"],
            "PAIRED_DROP_STABLE": bootstrap_summary[str(rate)]["PAIRED_DROP_STABLE"],
            "MULTI_CORPUS_TRUE_COUNT": multi_corpus[str(rate)]["true_count"],
            "MULTI_CORPUS_CONFIRMATION": multi_corpus[str(rate)]["MULTI_CORPUS_CONFIRMATION"],
            "FULL_NEW_RATE_BOUNDARY_EVENT": bool(structural and bootstrap_summary[str(rate)]["PAIRED_DROP_STABLE"] and multi_corpus[str(rate)]["MULTI_CORPUS_CONFIRMATION"]),
        }
    cr64 = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == 64)
    drop64 = endpoint_floor - cr64["recall"]
    repl64 = cr64["recall"] <= 0.50 and drop64 >= 0.25 and cr64["median_TRUE_MARGIN"] < 0 and cr64["fraction_wrong_centroid_closer"] >= 0.60 and bootstrap_summary["64"]["PAIRED_DROP_STABLE"]

    any_full = any(rate_events[str(r)]["FULL_NEW_RATE_BOUNDARY_EVENT"] for r in NEW_RATES)
    any_structural = any(rate_events[str(r)]["NEW_RATE_STRUCTURAL_EVENT"] for r in NEW_RATES)
    if domain_severe:
        final_classification = "DOMAIN_SHIFT_CONFOUNDED"
        subclass = ""
    elif not endpoint_pass:
        final_classification = "INCONCLUSIVE_CONFIRMATION"
        subclass = "ENDPOINT_BORDERLINE"
    elif any_full:
        final_classification = "PASS_NEW_RATE_BOUNDARY_CONFIRMED"
        subclass = ""
    elif repl64:
        final_classification = "PASS_DISCOVERY_RATE_REPLICATION_ONLY"
        subclass = ""
    elif any_structural:
        final_classification = "INCONCLUSIVE_CONFIRMATION"
        subclass = "PARTIAL_NEW_RATE_EVENT"
    else:
        final_classification = "NO_BOUNDARY_REPLICATION"
        subclass = ""

    if final_classification == "PASS_NEW_RATE_BOUNDARY_CONFIRMED":
        interpretation = "Under the frozen blind family-attribution probe, configuration-conditioned family-identifiability instability independently recurred on previously untouched source parents and at least one previously unseen operating-rate condition."
    elif final_classification == "PASS_DISCOVERY_RATE_REPLICATION_ONLY":
        interpretation = "The discovery-rate AAC instability replicated on untouched parents, but the pre-specified previously unseen rate probes did not provide sufficient support for a broader configuration-conditioned boundary."
    elif final_classification == "NO_BOUNDARY_REPLICATION":
        interpretation = "The prospectively frozen confirmation experiment did not reproduce the pre-specified configuration-boundary evidence."
    else:
        interpretation = "No stronger scientific interpretation is authorized by the prospective C1B gate."

    gate = {
        "ENDPOINT": {
            "BA_ENDPOINT_POOLED": endpoint_metrics["balanced_accuracy"],
            "AAC_RECALL_32": aac32,
            "AAC_RECALL_128": aac128,
            "AAC_ENDPOINT_RECALL_FLOOR": endpoint_floor,
            "ENDPOINT_CONTROL_PASS": bool(endpoint_pass),
            "DOMAIN_SHIFT_SEVERE": bool(domain_severe),
        },
        "48": rate_events["48"],
        "80": rate_events["80"],
        "96": rate_events["96"],
        "64": {
            "AAC_RECALL_64": cr64["recall"],
            "AAC_RECALL_DROP_64": drop64,
            "AAC_MEDIAN_TRUE_MARGIN_64": cr64["median_TRUE_MARGIN"],
            "AAC_WRONG_CENTROID_FRACTION_64": cr64["fraction_wrong_centroid_closer"],
            "PAIRED_DROP_BOOTSTRAP_LOW_64": bootstrap_summary["64"]["p2_5"],
            "PAIRED_DROP_STABLE_64": bootstrap_summary["64"]["PAIRED_DROP_STABLE"],
            "REPLICATION_64": bool(repl64),
        },
        "final_classification": final_classification,
        "interpretation_boundary": interpretation,
    }
    gate_path = ROOT / "30_results_c1b" / "gate" / "AQ2C1B_confirmation_gate.json"
    write_json(gate_path, gate)

    clean_lookup = {(r["derivative_uid"]): r for r in clean_distance_rows}
    traj_lookup = {(r["confirmation_parent_uid"], r["codec"], r["nominal_rate_kbps"]): r for r in trajectory_rows}
    per_aac_rate = {}
    for rate in RATES:
        items = [r for r in pred_rows if r["true_codec"] == "AAC" and r["nominal_rate_kbps"] == rate]
        cr = next(r for r in codec_rate_rows if r["codec"] == "AAC" and r["nominal_rate_kbps"] == rate)
        traj_items = [traj_lookup[(r["confirmation_parent_uid"], "AAC", rate)] for r in items if rate in [48, 64, 80, 96]]
        per_aac_rate[str(rate)] = {
            "recall": cr["recall"],
            "median_true_margin": cr["median_TRUE_MARGIN"],
            "fraction_true_centroid_rank_1": cr["fraction_true_centroid_rank_1"],
            "wrong_centroid_fraction": cr["fraction_wrong_centroid_closer"],
            "dominant_error_target": cr["dominant_error_target"],
            "dominant_error_share": cr["dominant_error_share"],
            "median_clean_distance": float(np.median([clean_lookup[r["derivative_uid"]]["STD_D_TO_CLEAN"] for r in items])),
            "median_trajectory_alpha": None if not traj_items else float(np.median([t["ALPHA"] for t in traj_items])),
            "fraction_within_endpoint_segment": None if not traj_items else float(np.mean([t["WITHIN_ENDPOINT_SEGMENT"] for t in traj_items])),
            "median_normalized_trajectory_residual": None if not traj_items else float(np.median([t["NORMALIZED_RESIDUAL"] for t in traj_items])),
        }
    confirmation_summary = {
        "confirmation_parent_count": 60,
        "parents_by_corpus": parents["source_corpus"].value_counts().to_dict(),
        "rate_grid": RATES,
        "codec_set": CLASS_ORDER,
        "per_rate": {str(r): {"balanced_accuracy": per_rate[str(r)]["balanced_accuracy"], "macro_f1": per_rate[str(r)]["macro_f1"], "AAC_recall": per_rate[str(r)]["per_codec_recall"]["AAC"], "MP3_recall": per_rate[str(r)]["per_codec_recall"]["MP3"], "Opus_recall": per_rate[str(r)]["per_codec_recall"]["Opus"]} for r in RATES},
        "per_AAC_rate": per_aac_rate,
        "bootstrap_summary": bootstrap_summary,
        "multi_corpus_summary": multi_corpus,
        "endpoint_gate": gate["ENDPOINT"],
        "new_rate_event_summary": {str(r): rate_events[str(r)] for r in NEW_RATES},
        "replication_64_summary": gate["64"],
        "final_classification": final_classification,
    }
    summary_path = ROOT / "30_results_c1b" / "gate" / "AQ2C1B_confirmation_summary.json"
    write_json(summary_path, confirmation_summary)

    artifact_paths = [
        derivative_manifest_path,
        ROOT / "27_confirmation_derivatives_c1b" / "AQ2C1B_derivative_file_sha256s.txt",
        collision_path,
        achieved_summary_path,
        feature_manifest_path,
        clean_feature_manifest_path,
        ROOT / "28_confirmation_features_c1b" / "AQ2C1B_feature_file_sha256s.txt",
        centroid_path,
        operational_spec_path,
        preconfirm_lock_path,
        centroid_audit_path,
        model_audit_path,
        pred_path,
        per_rate_path,
        per_codec_rate_path,
        bootstrap_summary_path,
        corpus_path,
        multi_corpus_path,
        clean_distance_path,
        trajectory_path,
        gate_path,
        summary_path,
        preprediction_path,
        ROOT / "logs_c1b" / "AQ2C1B_execution_script.py",
    ]
    for rate in BOOT_RATES:
        artifact_paths.append(ROOT / "30_results_c1b" / "bootstrap" / f"AQ2C1B_AAC_parent_bootstrap_{rate}.csv")
    artifact_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in artifact_paths if p.exists()}

    freeze = {
        "phase": "AQ-2C.1B",
        "specification_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": final_classification,
        "subclassification": subclass,
        "AQ2C1A_freeze_sha256": EXPECTED["AQ2C1A_FREEZE"],
        "prospective_confirmation_spec_sha256": EXPECTED["PROSPECTIVE_SPEC"],
        "AQ2C1B_preprediction_freeze_sha256": preprediction_sha,
        "preconfirm_operational_lock_sha256": sha256_file(preconfirm_lock_path),
        "confirmation_parent_manifest_sha256": EXPECTED["CONFIRM_PARENT_MANIFEST"],
        "confirmation_parent_count": 60,
        "parents_per_corpus": 20,
        "discovery_overlap_count": overlap,
        "discovery_pcm_collision_count": pcm_collision,
        "within_confirmation_pcm_duplicate_count": within_pcm_dup,
        "codec_set": CLASS_ORDER,
        "rate_grid": RATES,
        "new_probe_rates": NEW_RATES,
        "discovery_replication_rate": 64,
        "endpoint_rates": [32, 128],
        "expected_derivative_count": 1080,
        "actual_derivative_count": len(derivative_rows),
        "encode_success_count": 1080,
        "probe_success_count": 1080,
        "decode_success_count": 1080,
        "alignment_success_count": 1080,
        "feature_success_count": len(feature_rows),
        "clean_feature_success_count": len(clean_rows),
        "model_numeric_sha256": EXPECTED["MODEL_NUMERIC"],
        "model_config_sha256": EXPECTED["MODEL_CONFIG"],
        "centroid_numeric_sha256": sha256_file(centroid_path),
        "feature_spec_sha256": EXPECTED["FEATURE_SPEC"],
        "feature_extractor_sha256": EXPECTED["FEATURE_EXTRACTOR"],
        "model_refit_performed": "NO",
        "new_training_performed": "NO",
        "new_model_weights_created": "NO",
        "prediction_count": len(pred_rows),
        "endpoint_metrics": gate["ENDPOINT"],
        "per_rate_metrics": per_rate,
        "AAC_rate_profile": per_aac_rate,
        "parent_bootstrap_summary": bootstrap_summary,
        "multi_corpus_summary": multi_corpus,
        "new_rate_structural_events": {str(r): rate_events[str(r)]["NEW_RATE_STRUCTURAL_EVENT"] for r in NEW_RATES},
        "full_new_rate_boundary_events": {str(r): rate_events[str(r)]["FULL_NEW_RATE_BOUNDARY_EVENT"] for r in NEW_RATES},
        "replication_64": bool(repl64),
        "final_confirmation_gate": gate,
        "scientific_confirmation_result_produced": "YES",
        "vorbis_used": "NO",
        "representation_B_used": "NO",
        "AQ2A_v1_frozen_artifacts_modified": "NO",
        "AQ2A_R1_frozen_artifacts_modified": "NO",
        "AQ2A_R2_frozen_artifacts_modified": "NO",
        "AQ2B0_frozen_artifacts_modified": "NO",
        "AQ2B1_frozen_artifacts_modified": "NO",
        "AQ2B1_DX_frozen_artifacts_modified": "NO",
        "AQ2C1A_frozen_artifacts_modified": "NO",
        "existing_MM_artifacts_modified": "NO",
        "remaining_blockers": [],
        "artifact_hashes": artifact_hashes,
    }
    freeze_path = ROOT / "03_manifests" / "AQ2C1B_FREEZE.json"
    write_json(freeze_path, freeze)
    artifact_hashes[str(freeze_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(freeze_path)

    report_path = ROOT / "31_reports_c1b" / "AQ2C1B_EXECUTION_REPORT.md"
    sections = [
        ("1. Final C1B classification", final_classification),
        ("2. Subclassification", subclass),
        ("3. Maximum permitted scientific interpretation", interpretation),
        ("4. AQ-2C.1A freeze reconciliation", "PASS"),
        ("5. Prospective specification reconciliation", "PASS"),
        ("6. Confirmation-prediction-before-freeze audit", "PASS; prediction count was zero at PREPREDICTION freeze."),
        ("7. Confirmation parent identity", "60 untouched confirmation parents reconciled."),
        ("8. Parent counts by corpus", json.dumps(parents["source_corpus"].value_counts().to_dict(), indent=2)),
        ("9. Discovery overlap audit", str(overlap)),
        ("10. PCM collision audit", json.dumps({"discovery_pcm_collision": pcm_collision, "within_confirmation_duplicate": within_pcm_dup}, indent=2)),
        ("11. Frozen Python/toolchain reconciliation", "PASS"),
        ("12. Frozen B1 model reconciliation", "PASS"),
        ("13. Frozen feature pipeline reconciliation", "PASS"),
        ("14. B1 TRAIN centroid operational definition", json.dumps(operational_spec, indent=2)),
        ("15. Historical centroid reproduction result", "PASS"),
        ("16. Historical model prediction reproduction result", "PASS"),
        ("17. PRECONFIRM operational-lock SHA-256", sha256_file(preconfirm_lock_path)),
        ("18. Fresh-derivative requirement confirmation", "C1B derivatives generated under 27_confirmation_derivatives_c1b; C1A preflight derivatives not reused."),
        ("19. Expected versus actual derivatives", f"1080 / {len(derivative_rows)}"),
        ("20. Encode/probe/decode/alignment counts", "1080 / 1080 / 1080 / 1080"),
        ("21. Achieved bitrate engineering summary", "See 27_confirmation_derivatives_c1b/audit/AQ2C1B_achieved_rate_summary.csv."),
        ("22. Representation-A feature integrity", f"{len(feature_rows)} / 1080"),
        ("23. Clean-reference feature integrity", f"{len(clean_rows)} / 60"),
        ("24. PREPREDICTION FREEZE SHA-256", preprediction_sha),
        ("25. Explicit proof prediction count was zero at PREPREDICTION freeze", "confirmation_prediction_count = 0"),
        ("26. Frozen model application method", "z=(x-mean)/scale; logits=z@coef.T+intercept; argmax in AAC,MP3,Opus order; stable softmax for probabilities."),
        ("27. Per-rate balanced accuracy", json.dumps({r: per_rate[str(r)]["balanced_accuracy"] for r in RATES}, indent=2)),
        ("28. Per-rate macro-F1", json.dumps({r: per_rate[str(r)]["macro_f1"] for r in RATES}, indent=2)),
        ("29. Per-codec recall profile", json.dumps({r: per_rate[str(r)]["per_codec_recall"] for r in RATES}, indent=2)),
        ("30. Endpoint pooled BA", endpoint_metrics["balanced_accuracy"]),
        ("31. AAC recall 32", aac32),
        ("32. AAC recall 128", aac128),
        ("33. Endpoint-control gate", json.dumps(gate["ENDPOINT"], indent=2)),
        ("34. AAC endpoint recall floor", endpoint_floor),
        ("35. AAC 48-kbps criterion values", json.dumps(rate_events["48"], indent=2)),
        ("36. AAC 64-kbps criterion values", json.dumps(gate["64"], indent=2)),
        ("37. AAC 80-kbps criterion values", json.dumps(rate_events["80"], indent=2)),
        ("38. AAC 96-kbps criterion values", json.dumps(rate_events["96"], indent=2)),
        ("39. Parent-bootstrap 48 result", json.dumps(bootstrap_summary["48"], indent=2)),
        ("40. Parent-bootstrap 64 result", json.dumps(bootstrap_summary["64"], indent=2)),
        ("41. Parent-bootstrap 80 result", json.dumps(bootstrap_summary["80"], indent=2)),
        ("42. Parent-bootstrap 96 result", json.dumps(bootstrap_summary["96"], indent=2)),
        ("43. Multi-corpus 48 result", json.dumps(multi_corpus["48"], indent=2)),
        ("44. Multi-corpus 80 result", json.dumps(multi_corpus["80"], indent=2)),
        ("45. Multi-corpus 96 result", json.dumps(multi_corpus["96"], indent=2)),
        ("46. Full new-rate event status for 48/80/96", json.dumps({r: rate_events[str(r)]["FULL_NEW_RATE_BOUNDARY_EVENT"] for r in NEW_RATES}, indent=2)),
        ("47. 64-kbps replication status", repl64),
        ("48. Clean-reference distance summary", "See 30_results_c1b/geometry/AQ2C1B_clean_distance.csv."),
        ("49. Endpoint-trajectory geometry summary", "See 30_results_c1b/geometry/AQ2C1B_endpoint_trajectory_geometry.csv."),
        ("50. Final prospective classification calculation", json.dumps(gate, indent=2)),
        ("51. Whether independent new-rate support exists", any_full),
        ("52. Whether discovery-rate replication exists", repl64),
        ("53. Whether AQ-2C.2 was executed", "NO"),
        ("54. All generated artifacts", "\n".join(sorted(artifact_hashes))),
        ("55. Major SHA-256 values", json.dumps({**EXPECTED, **artifact_hashes}, indent=2)),
        ("56. Remaining blockers", "None"),
        ("57. Explicit statement", "AQ-2B.1 RESULT REMAINS =\nSTOP_MFEA1_TRACE_NOT_SUPPORTED"),
        ("58. Explicit statement", "AQ-2B.2 AUTHORIZED = NO"),
        ("59. Explicit statement", "CONFIRMATION PARENT COUNT = 60"),
        ("60. Explicit statement", "CONFIRMATION DERIVATIVES = 1080 / 1080"),
        ("61. Explicit statement", "NEW TRAINING PERFORMED = NO"),
        ("62. Explicit statement", "MODEL REFIT PERFORMED = NO"),
        ("63. Explicit statement", "NEW MODEL WEIGHTS CREATED = NO"),
        ("64. Explicit statement", "FROZEN B1 MODEL USED = YES"),
        ("65. Explicit statement", "VORBIS USED = NO"),
        ("66. Explicit statement", "REPRESENTATION B USED = NO"),
        ("67. Explicit statement", "AQ-2C.2 EXECUTED = NO"),
        ("68. Explicit statements", "AQ2A V1 FROZEN ARTIFACTS MODIFIED = NO\nAQ2A R1 FROZEN ARTIFACTS MODIFIED = NO\nAQ2A R2 FROZEN ARTIFACTS MODIFIED = NO\nAQ2B0 FROZEN ARTIFACTS MODIFIED = NO\nAQ2B1 FROZEN ARTIFACTS MODIFIED = NO\nAQ2B1-DX FROZEN ARTIFACTS MODIFIED = NO\nAQ2C1A FROZEN ARTIFACTS MODIFIED = NO\nEXISTING MM ARTIFACTS MODIFIED = NO"),
    ]
    lines = ["# AQ-2C.1B Execution Report", ""]
    for title, body in sections:
        lines.extend([f"## {title}", str(body), ""])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    artifact_hashes[str(report_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(report_path)

    command_log_path = ROOT / "logs_c1b" / "AQ2C1B_command_log.txt"
    command_log_path.write_text(json.dumps({"phase": "AQ-2C.1B", "created_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "cwd": str(ROOT), "environment": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]}, "final_classification": final_classification}, indent=2) + "\n", encoding="utf-8")
    artifact_hashes[str(command_log_path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(command_log_path)

    with open(ROOT / "AQ2C1B_SHA256SUMS.txt", "w", encoding="utf-8", newline="\n") as f:
        for rel, h in sorted(artifact_hashes.items()):
            f.write(f"{h}  {rel}\n")

    print(json.dumps({
        "final_classification": final_classification,
        "subclassification": subclass,
        "prediction_count": len(pred_rows),
        "BA_ENDPOINT_POOLED": endpoint_metrics["balanced_accuracy"],
        "AAC_RECALL_32": aac32,
        "AAC_RECALL_128": aac128,
        "full_new_rate_boundary_events": {r: rate_events[str(r)]["FULL_NEW_RATE_BOUNDARY_EVENT"] for r in NEW_RATES},
        "replication_64": repl64,
        "ledger_sha256": sha256_file(ROOT / "AQ2C1B_SHA256SUMS.txt"),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
