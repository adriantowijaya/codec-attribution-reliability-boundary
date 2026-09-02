import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.fftpack import dct
from scipy.signal import correlate
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(r"[REDACTED_LOCAL_PATH]\anaconda3\python.exe")
ENCODER = ROOT / "tools/gyan_ffmpeg_9.0_full/bin/ffmpeg.exe"
FFPROBE = ROOT / "tools/gyan_ffmpeg_9.0_full/bin/ffprobe.exe"
DECODER = Path(r"[REDACTED_LOCAL_PATH]\AppData\Local\CapCut\Apps\7.7.0.3143\ffmpeg.exe")
ARCHIVE = ROOT / "39_bh_speech/source_archive/test-clean.tar.gz"
LIBRI_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LIBRI_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
CLASS_ORDER = ["AAC", "MP3", "Opus"]
RATES = [32, 80, 96, 128]
SEGMENT_SECONDS = 2.0
SEGMENT_SAMPLES = 96000
ALIGN_BOUND = 8192


EXPECTED = {
    "03_manifests/AQ2C1B_FREEZE.json": "c7dfd6a06e32616fd9396e827af1f170e90f421c2b1f6d2a2997f3aa0fd86579",
    "03_manifests/AQ2C1B_PREPREDICTION_FREEZE.json": "a9a19c65abbeed2dd8bcd7c1fdd8815511f10baa384a192d8132429c5e290c79",
    "03_manifests/AQ2C2_FREEZE.json": "e658519364f6236251f68deb2b3a87ee73a41e49f49e5621faf1edcdaa9b194e",
    "03_manifests/AQ2C2_DX_FREEZE.json": "7b76453ee85d0c1e0d424fbb19880585efc04a54ed0b4638960bd72ed29329b5",
    "37_diagnostics_c2_dx/gate/AQ2C2_DX_diagnostic_gate.json": "6975e915067ee0fd6a107f69d9c011a9e6accb66bfbde1fbb0ff751a67ac30de",
    "18_model_b1/AQ2B1_model_numeric.npz": "7850e14a7231ac5d2d06a72c6c2eb487cedd7df32cf5750bf62ddda79932b5de",
    "18_model_b1/AQ2B1_model_config.json": "495b976b5fe558faee6a374707e5629c04a8d8b844f5aea9d178259db5d1fdff",
    "14_feature_preflight_b0/AQ2B0_feature_spec.json": "83020402082b7168d88df8d1eb02fc4f254d24b6fbd4ab0770403d1f96c1acae",
    "14_feature_preflight_b0/aq2b0_feature_extractor.py": "6c79cbc06c67c9cff140cd64de6f878ea59f1b1580bb7db0b0acb08a4a57da8f",
    "16_derivatives_b1/AQ2B1_derivative_manifest.csv": "6f32143b4e0267e3f2acdf91404954a9ddb1e2ad5c5d5bf6cb99651dc9e021aa",
    "17_features_b1/AQ2B1_feature_manifest.csv": "e6e9983484bea4d44756a7b9b71bbea85a687fbd07892cdd42ff9fcbbec74754",
    str(ENCODER): "57c56e369d5b4873b4d93fc1a1d833cb7cd8bc9325c14b05c34ce60b22842d8a",
    str(FFPROBE): "afe05347caaabe479b3c4eae71992b6ec1e11c57266a1d665deb0f9fe9847208",
    str(DECODER): "4dc3e63209cb6f183b703c8842f6e3dcc22778ccca1a3b9f4b5fca4034bb54dd",
}


RF_CONFIG = {
    "n_estimators": 500, "criterion": "gini", "max_depth": None, "min_samples_split": 2,
    "min_samples_leaf": 1, "min_weight_fraction_leaf": 0.0, "max_features": "sqrt",
    "max_leaf_nodes": None, "min_impurity_decrease": 0.0, "bootstrap": True,
    "oob_score": False, "n_jobs": 1, "random_state": 20261101, "verbose": 0,
    "warm_start": False, "class_weight": None, "ccp_alpha": 0.0, "max_samples": None,
}


def mkdirs():
    for rel in [
        "39_bh_speech/source_archive", "39_bh_speech/extracted", "39_bh_speech/inventory",
        "39_bh_speech/selection_audit", "39_bh_speech/canonical_selected", "39_bh_speech/license",
        "40_bh_models/rf", "40_bh_models/mfcc_lr", "40_bh_models/validation", "40_bh_models/freeze",
        "41_bh_representation/mfcc_spec", "41_bh_representation/mfcc_extractor",
        "41_bh_representation/training_features", "41_bh_representation/validation_features",
        "41_bh_representation/speech_features", "41_bh_representation/determinism",
        "41_bh_representation/repA_speech_features", "42_bh_derivatives/encoded",
        "42_bh_derivatives/decoded_raw", "42_bh_derivatives/aligned_pcm", "42_bh_derivatives/audit",
        "42_bh_derivatives/scratch", "43_bh_results/primary", "43_bh_results/rf",
        "43_bh_results/mfcc_lr", "43_bh_results/bootstrap", "43_bh_results/comparison",
        "43_bh_results/gate", "44_bh_reports", "logs_bh",
    ]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_cmd(cmd, timeout=180):
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
    return {"command": " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd), "return_code": cp.returncode, "stdout": cp.stdout, "stderr": cp.stderr}


def fail(classification, subclassification, details):
    mkdirs()
    freeze = {
        "phase": "AQ-2D.0-BH",
        "specification_version": "v1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": classification,
        "subclassification": subclassification,
        "remaining_blockers": details,
    }
    write_json(ROOT / "03_manifests/AQ2D0BH_FREEZE.json", freeze)
    raise SystemExit(json.dumps(freeze, indent=2, allow_nan=False))


def verify_predecessors():
    audit = {"hashes": {}, "field_checks": {}, "environment": {}}
    for key, expected in EXPECTED.items():
        path = Path(key) if Path(key).is_absolute() else ROOT / key
        actual = sha256_file(path)
        audit["hashes"][key] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            write_json(ROOT / "40_bh_models/validation/AQ2D0BH_predecessor_identity_audit.json", audit)
            fail("FAIL_INTEGRITY", "PREDECESSOR_HASH_MISMATCH", audit)
    c2dx = json.loads((ROOT / "03_manifests/AQ2C2_DX_FREEZE.json").read_text(encoding="utf-8"))
    c2dx_gate = json.loads((ROOT / "37_diagnostics_c2_dx/gate/AQ2C2_DX_diagnostic_gate.json").read_text(encoding="utf-8"))
    checks = {
        "C2DX_final": c2dx["final_classification"] == "DX_INCONCLUSIVE",
        "ONE_PROBE_ONLY_SUCCESSOR_AUTHORIZED": c2dx_gate["ONE_PROBE_ONLY_SUCCESSOR_AUTHORIZED"] == "NO",
        "COHORT_FORENSICS_AUTHORIZED": c2dx_gate["COHORT_FORENSICS_AUTHORIZED"] == "NO",
        "C1B_result_changed": c2dx_gate["C1B_result_changed"] == "NO",
        "C2_result_changed": c2dx_gate["C2_result_changed"] == "NO",
    }
    audit["field_checks"] = checks
    audit["environment"] = {
        "python_executable": str(PYTHON),
        "python_sha256": sha256_file(PYTHON),
        "python_version": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "thread_env": {k: os.environ.get(k) for k in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
    }
    if audit["environment"]["python_sha256"] != "62c225fb9cdc41b139c7024581c233644f975ffc35314558c60ebefa6b88be01":
        fail("FAIL_INTEGRITY", "PYTHON_HASH_MISMATCH", audit)
    if (platform.python_version(), np.__version__, scipy.__version__, pd.__version__, sklearn.__version__) != ("3.12.3", "1.26.4", "1.13.1", "2.2.2", "1.5.1"):
        fail("FAIL_INTEGRITY", "PYTHON_ENV_MISMATCH", audit)
    if not all(checks.values()):
        fail("FAIL_INTEGRITY", "C2_DX_FIELD_MISMATCH", audit)
    write_json(ROOT / "40_bh_models/validation/AQ2D0BH_predecessor_identity_audit.json", audit)


def ffprobe_audio(path):
    cmd = [str(FFPROBE), "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels,duration,bit_rate:format=format_name,duration,bit_rate", "-of", "json", str(path)]
    r = run_cmd(cmd, timeout=60)
    if r["return_code"] != 0:
        return None, r
    try:
        js = json.loads(r["stdout"])
    except json.JSONDecodeError:
        return None, r
    stream = js.get("streams", [{}])[0] if js.get("streams") else {}
    fmt = js.get("format", {})
    dur = stream.get("duration") or fmt.get("duration") or 0
    return {
        "codec_name": stream.get("codec_name", ""),
        "sample_rate": int(stream.get("sample_rate", 0)),
        "channels": int(stream.get("channels", 0)),
        "duration": float(dur),
        "format_name": fmt.get("format_name", ""),
        "bit_rate": stream.get("bit_rate") or fmt.get("bit_rate") or "",
    }, r


def acquire_librispeech():
    if not ARCHIVE.exists():
        tmp = ARCHIVE.with_suffix(".download")
        try:
            urllib.request.urlretrieve(LIBRI_URL, tmp)
            tmp.replace(ARCHIVE)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            fail("BLOCKED_SPEECH_CORPUS_ACQUISITION", "CANONICAL_DOWNLOAD_FAILED", {"url": LIBRI_URL, "error": repr(exc)})
    md5 = md5_file(ARCHIVE)
    if md5 != LIBRI_MD5:
        fail("FAIL_INTEGRITY", "LIBRISPEECH_MD5_MISMATCH", {"expected": LIBRI_MD5, "actual": md5, "path": str(ARCHIVE)})
    record = {
        "external_corpus": "LibriSpeech ASR corpus",
        "resource": "OpenSLR SLR12",
        "subset": "test-clean",
        "canonical_source_url": LIBRI_URL,
        "official_archive_name": "test-clean.tar.gz",
        "official_expected_md5": LIBRI_MD5,
        "observed_md5": md5,
        "archive_sha256": sha256_file(ARCHIVE),
        "file_size_bytes": ARCHIVE.stat().st_size,
        "license": "CC BY 4.0",
    }
    write_json(ROOT / "39_bh_speech/license/AQ2D0BH_LibriSpeech_source_record.json", record)
    return record


def extract_archive():
    marker = ROOT / "39_bh_speech/extracted/LibriSpeech/test-clean"
    if marker.exists():
        return marker
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        base = (ROOT / "39_bh_speech/extracted").resolve()
        for member in tf.getmembers():
            target = (base / member.name).resolve()
            if not str(target).startswith(str(base)):
                fail("FAIL_INTEGRITY", "TAR_PATH_TRAVERSAL", {"member": member.name})
        tf.extractall(base)
    if not marker.exists():
        fail("FAIL_INTEGRITY", "LIBRISPEECH_EXTRACT_STRUCTURE_MISMATCH", {"expected": str(marker)})
    return marker


def inventory_librispeech(test_clean_dir):
    rows = []
    for flac in sorted(test_clean_dir.rglob("*.flac")):
        stem = flac.stem
        parts = stem.split("-")
        if len(parts) != 3:
            fail("FAIL_INTEGRITY", "LIBRISPEECH_FILENAME_PATTERN_MISMATCH", {"path": str(flac)})
        info, _ = ffprobe_audio(flac)
        if info is None:
            fail("FAIL_INTEGRITY", "LIBRISPEECH_FFPROBE_FAILED", {"path": str(flac)})
        rows.append({
            "speaker_id": parts[0],
            "chapter_id": parts[1],
            "utterance_id": parts[2],
            "absolute_path": str(flac),
            "file_size": flac.stat().st_size,
            "source_file_sha256": sha256_file(flac),
            "sample_rate": info["sample_rate"],
            "channels": info["channels"],
            "duration": info["duration"],
        })
    write_csv(ROOT / "39_bh_speech/inventory/AQ2D0BH_LibriSpeech_test_clean_inventory.csv", rows)
    speakers = {r["speaker_id"] for r in rows}
    if len(speakers) != 40:
        fail("FAIL_INTEGRITY", "LIBRISPEECH_SPLIT_IDENTITY_MISMATCH", {"utterances": len(rows), "unique_speaker_count": len(speakers)})
    return rows


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


def raw_pcm_sha256(path):
    samples, info = wav_read(path)
    if info != {"sample_rate": 48000, "channels": 1, "sample_width": 2, "sample_count": 96000}:
        fail("FAIL_INTEGRITY", "PCM_CONTRACT_MISMATCH", {"path": str(path), "info": info})
    return hashlib.sha256(samples.astype("<i2", copy=False).tobytes()).hexdigest()


def canonicalize_flac(flac_path, out_path):
    tmp = ROOT / "42_bh_derivatives/scratch" / f"{Path(flac_path).stem}.decode.wav"
    r = run_cmd([str(DECODER), "-y", "-hide_banner", "-i", str(flac_path), "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(tmp)], timeout=120)
    if r["return_code"] != 0 or not tmp.exists():
        return False, r, {}
    samples, info = wav_read(tmp)
    if info["sample_rate"] != 48000 or info["channels"] != 1 or info["sample_width"] != 2 or len(samples) < SEGMENT_SAMPLES:
        return False, r, info
    start = (len(samples) - SEGMENT_SAMPLES) // 2
    wav_write(out_path, samples[start:start + SEGMENT_SAMPLES])
    return True, r, {"decoded_samples": int(len(samples)), "center_crop_start": int(start)}


def select_speech_parents(inventory):
    by_speaker = defaultdict(list)
    for r in inventory:
        score = sha256_text(f"AQ2D0BH_SPEECH_SELECTION_V1|{r['speaker_id']}|{r['utterance_id']}")
        rr = dict(r)
        rr["selection_score"] = score
        by_speaker[r["speaker_id"]].append(rr)
    selected, audit_rows, seen_pcm = [], [], set()
    for speaker in sorted(by_speaker):
        chosen = None
        for cand in sorted(by_speaker[speaker], key=lambda x: (x["selection_score"], x["utterance_id"])):
            reason = ""
            if not Path(cand["absolute_path"]).exists():
                reason = "SOURCE_FLAC_MISSING"
            elif float(cand["duration"]) < 2.0:
                reason = "DURATION_LT_2"
            elif int(cand["channels"]) != 1:
                reason = "NOT_MONO"
            else:
                parent_uid = sha256_text(f"AQ2D0BH_SPEECH_PARENT_V1|{speaker}|{cand['chapter_id']}|{cand['utterance_id']}|{cand['source_file_sha256']}")[:32]
                out = ROOT / "39_bh_speech/canonical_selected" / f"{speaker}-{cand['chapter_id']}-{cand['utterance_id']}.wav"
                ok, cmd, extra = canonicalize_flac(cand["absolute_path"], out)
                if not ok:
                    reason = "CANONICALIZATION_FAILED"
                else:
                    pcm_sha = raw_pcm_sha256(out)
                    if pcm_sha in seen_pcm:
                        reason = "DUPLICATE_CANONICAL_PCM"
                    else:
                        seen_pcm.add(pcm_sha)
                        chosen = {
                            "speech_parent_uid": parent_uid,
                            "speaker_id": speaker,
                            "chapter_id": cand["chapter_id"],
                            "utterance_id": cand["utterance_id"],
                            "selection_score": cand["selection_score"],
                            "source_flac_path": cand["absolute_path"],
                            "source_file_sha256": cand["source_file_sha256"],
                            "source_duration": cand["duration"],
                            "canonical_pcm_path": str(out),
                            "canonical_raw_pcm_sha256": pcm_sha,
                            "canonical_sample_rate": 48000,
                            "canonical_channels": 1,
                            "canonical_sample_width": 2,
                            "canonical_sample_count": 96000,
                            **extra,
                        }
                        reason = "SELECTED"
            audit_rows.append({"speaker_id": speaker, "chapter_id": cand["chapter_id"], "utterance_id": cand["utterance_id"], "selection_score": cand["selection_score"], "decision": reason})
            if chosen:
                selected.append(chosen)
                break
        if not chosen:
            write_json(ROOT / "39_bh_speech/selection_audit/AQ2D0BH_speech_selection_audit.json", {"rows": audit_rows, "selected_count": len(selected)})
            fail("BLOCKED_SPEECH_COHORT", "NO_ELIGIBLE_UTTERANCE_FOR_SPEAKER", {"speaker_id": speaker})
    if len(selected) != 40 or len({s["speaker_id"] for s in selected}) != 40 or len({s["canonical_raw_pcm_sha256"] for s in selected}) != 40:
        fail("BLOCKED_SPEECH_COHORT", "SPEECH_PARENT_COUNT_MISMATCH", {"selected": len(selected)})
    write_json(ROOT / "39_bh_speech/selection_audit/AQ2D0BH_speech_selection_audit.json", {"selection_rule": "SHA256(AQ2D0BH_SPEECH_SELECTION_V1|speaker_id|utterance_id), first eligible per speaker", "rows": audit_rows, "selected_count": len(selected)})
    write_csv(ROOT / "39_bh_speech/AQ2D0BH_speech_parent_manifest.csv", selected)
    return selected


def load_rep_a():
    spec = json.loads((ROOT / "14_feature_preflight_b0/AQ2B0_feature_spec.json").read_text(encoding="utf-8"))
    module_path = ROOT / "14_feature_preflight_b0/aq2b0_feature_extractor.py"
    module_spec = importlib.util.spec_from_file_location("aq2b0_feature_extractor", module_path)
    mod = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(mod)
    return spec, mod


def rep_a_from_wav(path, spec, mod):
    x = mod.read_pcm16_wav(path)
    return mod.extract_representation_a(mod.extract_stft_core(x, spec), spec)


def canonical_array_hash(arr, representation_id):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    shape = ",".join(str(x) for x in arr.shape)
    payload = representation_id.encode("ascii") + b"\n" + shape.encode("ascii") + b"\n<f8\n" + arr.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def load_b1():
    deriv = pd.read_csv(ROOT / "16_derivatives_b1/AQ2B1_derivative_manifest.csv")
    feats = pd.read_csv(ROOT / "17_features_b1/AQ2B1_feature_manifest.csv")
    fmap = dict(zip(feats["derivative_uid"], feats["npy_path"]))
    train = deriv[(deriv["split"] == "TRAIN") & (deriv["nominal_rate_kbps"].isin([32, 128]))].copy()
    val = deriv[(deriv["split"] == "VALIDATION") & (deriv["nominal_rate_kbps"].isin([32, 128]))].copy()
    if len(train) != 294 or train["parent_uid"].nunique() != 49 or len(val) != 96 or val["parent_uid"].nunique() != 16:
        fail("FAIL_INTEGRITY", "B1_SPLIT_COUNT_MISMATCH", {"train_rows": len(train), "train_parents": int(train["parent_uid"].nunique()), "val_rows": len(val), "val_parents": int(val["parent_uid"].nunique())})
    return deriv, fmap, train, val


def fit_rf(train, fmap):
    model_npz = np.load(ROOT / "18_model_b1/AQ2B1_model_numeric.npz", allow_pickle=False)
    X = np.vstack([np.load(fmap[u], allow_pickle=False) for u in train["derivative_uid"]])
    Z = (X - model_npz["scaler_mean"]) / model_npz["scaler_scale"]
    y = train["codec_family"].to_numpy()
    rf = RandomForestClassifier(**RF_CONFIG)
    rf.fit(Z, y)
    out = ROOT / "40_bh_models/rf/AQ2D0BH_RF_model.joblib"
    joblib.dump(rf, out)
    cfg = {"classifier": {"type": "RandomForestClassifier", **RF_CONFIG}, "fit_scope": "B1_TRAIN_STANDARDIZED_REPRESENTATION_A_ONLY", "training_rows": 294, "training_parents": 49, "sklearn_version": sklearn.__version__, "class_order": CLASS_ORDER}
    write_json(ROOT / "40_bh_models/rf/AQ2D0BH_RF_config.json", cfg)
    return rf, sha256_file(out)


def mfcc_spec_obj():
    sr, frame, hop, n_fft, n_mels = 48000, 1200, 480, 2048, 40
    hann = 0.5 - 0.5 * np.cos((2 * np.pi * np.arange(frame, dtype=np.float64)) / (frame - 1))
    mel_min = 2595 * np.log10(1 + 0 / 700)
    mel_max = 2595 * np.log10(1 + 24000 / 700)
    mel_edges = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_edges = 700 * (10 ** (mel_edges / 2595) - 1)
    freqs = np.fft.rfftfreq(n_fft, d=1 / sr)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        up = (freqs - left) / (center - left)
        down = (right - freqs) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return {
        "representation_id": "MFCC_STATISTICS_V1", "sample_rate": sr, "sample_count": 96000,
        "frame_length": frame, "hop_length": hop, "n_fft": n_fft, "frame_count": 198,
        "n_mels": n_mels, "fmin": 0, "fmax": 24000, "mel_formula": "2595*log10(1+f/700)",
        "dct": {"type": "II", "norm": "ortho", "coefficients_retained": 20},
        "statistics": ["temporal_mean", "temporal_std_ddof0"], "dimension": 40,
        "hann_sha256": hashlib.sha256(np.ascontiguousarray(hann.astype("<f8")).tobytes()).hexdigest(),
        "mel_filterbank_sha256": hashlib.sha256(np.ascontiguousarray(fb.astype("<f8")).tobytes()).hexdigest(),
    }, hann, fb


MFCC_EXTRACTOR_SOURCE = r'''import hashlib
import json
import wave
from pathlib import Path

import numpy as np
from scipy.fftpack import dct

REPRESENTATION_ID = "MFCC_STATISTICS_V1"

def read_pcm16_wav(path):
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        data = w.readframes(n)
    if sr != 48000 or ch != 1 or sw != 2 or n != 96000:
        raise ValueError(f"invalid PCM contract sr={sr} ch={ch} sw={sw} n={n}")
    return np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0

def symmetric_hann(n):
    i = np.arange(n, dtype=np.float64)
    return 0.5 - 0.5 * np.cos((2.0 * np.pi * i) / float(n - 1))

def mel_filterbank(sr=48000, n_fft=2048, n_mels=40, fmin=0.0, fmax=24000.0):
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mel_edges = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_edges = 700.0 * (10.0 ** (mel_edges / 2595.0) - 1.0)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    fb = np.zeros((n_mels, len(freqs)), dtype=np.float64)
    for m in range(n_mels):
        left, center, right = hz_edges[m], hz_edges[m + 1], hz_edges[m + 2]
        up = (freqs - left) / (center - left)
        down = (right - freqs) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb

def extract_from_pcm_float(x):
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (96000,):
        raise ValueError(f"invalid input shape {x.shape}")
    frame, hop, n_fft, frames = 1200, 480, 2048, 198
    win = symmetric_hann(frame)
    win_power = float(np.sum(win * win))
    fb = mel_filterbank()
    mfcc = np.empty((20, frames), dtype=np.float64)
    for t in range(frames):
        start = t * hop
        segment = x[start:start + frame]
        if segment.shape[0] != frame:
            raise ValueError("incomplete frame")
        spec = np.fft.rfft(segment * win, n=n_fft)
        power = (np.abs(spec) ** 2) / win_power
        logmel = np.log(np.maximum(fb @ power, 1e-12))
        mfcc[:, t] = dct(logmel, type=2, norm="ortho")[:20]
    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1, ddof=0)]).astype(np.float64, copy=False)
    if feat.shape != (40,) or not np.all(np.isfinite(feat)):
        raise ValueError("invalid MFCC feature")
    return feat

def extract_from_wav(path):
    return extract_from_pcm_float(read_pcm16_wav(path))

def canonical_array_hash(arr):
    arr = np.ascontiguousarray(np.asarray(arr, dtype="<f8"))
    payload = REPRESENTATION_ID.encode("ascii") + b"\n" + ",".join(str(x) for x in arr.shape).encode("ascii") + b"\n<f8\n" + arr.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()
'''


def write_mfcc_representation():
    spec, _, _ = mfcc_spec_obj()
    spec_path = ROOT / "41_bh_representation/mfcc_spec/AQ2D0BH_MFCC_STATISTICS_V1.json"
    ext_path = ROOT / "41_bh_representation/mfcc_extractor/aq2d0bh_mfcc_extractor.py"
    write_json(spec_path, spec)
    ext_path.write_text(MFCC_EXTRACTOR_SOURCE, encoding="utf-8")
    return spec_path, ext_path


def extract_mfcc(path):
    spec, hann, fb = mfcc_spec_obj()
    samples, info = wav_read(path)
    if info != {"sample_rate": 48000, "channels": 1, "sample_width": 2, "sample_count": 96000}:
        raise ValueError(f"invalid PCM {path} {info}")
    x = samples.astype(np.float64) / 32768.0
    win_power = float(np.sum(hann * hann))
    features = np.empty((20, 198), dtype=np.float64)
    for t in range(198):
        frame = x[t * 480:t * 480 + 1200]
        spectrum = np.fft.rfft(frame * hann, n=2048)
        power = (np.abs(spectrum) ** 2) / win_power
        logmel = np.log(np.maximum(fb @ power, 1e-12))
        features[:, t] = dct(logmel, type=2, norm="ortho")[:20]
    out = np.concatenate([np.mean(features, axis=1), np.std(features, axis=1, ddof=0)]).astype(np.float64, copy=False)
    if out.shape != (40,) or not np.all(np.isfinite(out)):
        raise ValueError("bad MFCC feature")
    return out


def mfcc_determinism_and_fit(train, val):
    write_mfcc_representation()
    runs = []
    for run in [1, 2, 3]:
        out_dir = ROOT / f"41_bh_representation/determinism/RUN_{run}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for _, row in train.sort_values("derivative_uid").iterrows():
            arr = extract_mfcc(Path(row["aligned_pcm_path"]))
            npy = out_dir / f"{row['derivative_uid']}.npy"
            np.save(npy, arr, allow_pickle=False)
            rows.append({"derivative_uid": row["derivative_uid"], "npy_path": str(npy), "array_sha256": canonical_array_hash(arr, "MFCC_STATISTICS_V1"), "npy_file_sha256": sha256_file(npy), "shape": "40", "dtype": "float64", "finite_check": True})
        runs.append(rows)
    det_rows = []
    for idx in range(len(runs[0])):
        uid = runs[0][idx]["derivative_uid"]
        a = np.load(runs[0][idx]["npy_path"], allow_pickle=False)
        for comp in [1, 2]:
            b = np.load(runs[comp][idx]["npy_path"], allow_pickle=False)
            det_rows.append({"derivative_uid": uid, "comparison": f"RUN_1_vs_RUN_{comp+1}", "array_equal": bool(np.array_equal(a, b)), "max_abs_diff": float(np.max(np.abs(a - b))), "hash_equal": runs[0][idx]["array_sha256"] == runs[comp][idx]["array_sha256"]})
    write_csv(ROOT / "41_bh_representation/determinism/AQ2D0BH_MFCC_determinism.csv", det_rows)
    if not all(r["array_equal"] and r["max_abs_diff"] == 0 and r["hash_equal"] for r in det_rows):
        fail("FAIL_INTEGRITY", "MFCC_REPRESENTATION_NONDETERMINISTIC", det_rows[:5])
    X = np.vstack([np.load(r["npy_path"], allow_pickle=False) for r in runs[0]])
    y = train.sort_values("derivative_uid")["codec_family"].to_numpy()
    scaler = StandardScaler(with_mean=True, with_std=True)
    Z = scaler.fit_transform(X)
    lr = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, tol=1e-8, multi_class="multinomial")
    lr.fit(Z, y)
    class_order = [str(x) for x in lr.classes_]
    if class_order != CLASS_ORDER:
        fail("FAIL_INTEGRITY", "MFCC_CLASS_ORDER_MISMATCH", class_order)
    npz = ROOT / "40_bh_models/mfcc_lr/AQ2D0BH_MFCC_LR_numeric.npz"
    np.savez(npz, scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_, model_coef=lr.coef_, model_intercept=lr.intercept_, class_order=np.asarray(CLASS_ORDER))
    cfg = {"representation_id": "MFCC_STATISTICS_V1", "scaler": {"type": "StandardScaler", "with_mean": True, "with_std": True}, "model": {"type": "LogisticRegression", "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 5000, "tol": 1e-8, "multi_class": "multinomial"}, "fit_scope": "B1_TRAIN_MFCC_ONLY", "training_rows": 294, "training_parents": 49, "class_order": CLASS_ORDER, "sklearn_version": sklearn.__version__}
    write_json(ROOT / "40_bh_models/mfcc_lr/AQ2D0BH_MFCC_LR_config.json", cfg)
    centroids = {c: np.mean(Z[y == c], axis=0) for c in CLASS_ORDER}
    cent_path = ROOT / "40_bh_models/mfcc_lr/AQ2D0BH_MFCC_train_centroids.npz"
    np.savez(cent_path, centroid_AAC=centroids["AAC"], centroid_MP3=centroids["MP3"], centroid_Opus=centroids["Opus"], class_order=np.asarray(CLASS_ORDER))
    val_rows = []
    for _, row in val.sort_values("derivative_uid").iterrows():
        out = ROOT / "41_bh_representation/validation_features" / f"{row['derivative_uid']}.npy"
        arr = extract_mfcc(Path(row["aligned_pcm_path"]))
        np.save(out, arr, allow_pickle=False)
        val_rows.append({"derivative_uid": row["derivative_uid"], "codec_family": row["codec_family"], "nominal_rate_kbps": int(row["nominal_rate_kbps"]), "feature_path": str(out)})
    preds = []
    for r in val_rows:
        x = np.load(r["feature_path"], allow_pickle=False)
        z = (x - scaler.mean_) / scaler.scale_
        logits = z @ lr.coef_.T + lr.intercept_
        preds.append({**r, "predicted_codec": CLASS_ORDER[int(np.argmax(logits))]})
    metrics = endpoint_metrics(preds, "predicted_codec")
    metrics["MFCC_VALIDATION_COMPETENT"] = endpoint_pass(metrics)
    write_json(ROOT / "40_bh_models/validation/AQ2D0BH_MFCC_validation.json", metrics)
    return {"numeric_sha256": sha256_file(npz), "centroid_sha256": sha256_file(cent_path), "validation": metrics, "authorized": "YES" if metrics["MFCC_VALIDATION_COMPETENT"] else "NO", "scaler": scaler, "lr": lr, "centroids": centroids}


def endpoint_metrics(rows, pred_key):
    endpoint = [r for r in rows if int(r["nominal_rate_kbps"]) in [32, 128]]
    y_true = [r["codec_family"] if "codec_family" in r else r["true_codec"] for r in endpoint]
    y_pred = [r[pred_key] for r in endpoint]
    recalls = {}
    for c in CLASS_ORDER:
        denom = sum(1 for y in y_true if y == c)
        recalls[c] = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c) / denom if denom else 0.0
    def aac(rate):
        sub = [r for r in rows if (r["codec_family"] if "codec_family" in r else r["true_codec"]) == "AAC" and int(r["nominal_rate_kbps"]) == rate]
        return sum(1 for r in sub if r[pred_key] == "AAC") / len(sub) if sub else 0.0
    return {"BA": float(np.mean([recalls[c] for c in CLASS_ORDER])), "per_codec_recall": recalls, "AAC_RECALL_32": aac(32), "AAC_RECALL_128": aac(128)}


def endpoint_pass(m):
    return bool(m["BA"] >= 0.75 and m["AAC_RECALL_32"] >= 0.60 and m["AAC_RECALL_128"] >= 0.60)


def endpoint_severe(m):
    return bool(m["BA"] < 0.65 or (m["AAC_RECALL_32"] < 0.50 and m["AAC_RECALL_128"] < 0.50))


def validate_rf(rf, val, fmap):
    model_npz = np.load(ROOT / "18_model_b1/AQ2B1_model_numeric.npz", allow_pickle=False)
    preds = []
    for _, row in val.iterrows():
        x = np.load(fmap[row["derivative_uid"]], allow_pickle=False)
        z = (x - model_npz["scaler_mean"]) / model_npz["scaler_scale"]
        pred = rf.predict(z.reshape(1, -1))[0]
        preds.append({"codec_family": row["codec_family"], "nominal_rate_kbps": int(row["nominal_rate_kbps"]), "predicted_codec": pred})
    metrics = endpoint_metrics(preds, "predicted_codec")
    metrics["RF_VALIDATION_COMPETENT"] = endpoint_pass(metrics)
    write_json(ROOT / "40_bh_models/validation/AQ2D0BH_RF_validation.json", metrics)
    return metrics


def codec_info(codec, rate):
    if codec == "AAC":
        return {"ext": "m4a", "probe": "aac", "encoder": "aac", "core": "FFmpeg-native AAC", "args": ["-c:a", "aac", "-b:a", f"{rate}k"]}
    if codec == "MP3":
        return {"ext": "mp3", "probe": "mp3", "encoder": "libmp3lame", "core": "libmp3lame", "args": ["-c:a", "libmp3lame", "-b:a", f"{rate}k"]}
    if codec == "Opus":
        return {"ext": "opus", "probe": "opus", "encoder": "libopus", "core": "libopus", "args": ["-c:a", "libopus", "-b:a", f"{rate}k", "-vbr", "off"]}
    raise ValueError(codec)


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


def design_and_arm_freezes(record, parents, rf_sha, rf_val, mfcc_state):
    design = {
        "phase": "AQ-2D.0-BH", "specification_version": "v1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_corpus": "LibriSpeech test-clean", "archive_md5": record["observed_md5"], "archive_sha256": record["archive_sha256"],
        "license": "CC BY 4.0", "speaker_count": 40, "speech_parent_count": 40,
        "speech_parent_manifest_sha256": sha256_file(ROOT / "39_bh_speech/AQ2D0BH_speech_parent_manifest.csv"),
        "codec_set": CLASS_ORDER, "rate_grid": RATES, "expected_derivatives": 480,
        "primary_arm_spec": "PRIMARY_A_LR: frozen STFT_LOGPOWER_STATS_V1 + frozen B1 StandardScaler + frozen multinomial LogisticRegression",
        "RF_exact_config": RF_CONFIG,
        "MFCC_exact_representation_spec": json.loads((ROOT / "41_bh_representation/mfcc_spec/AQ2D0BH_MFCC_STATISTICS_V1.json").read_text(encoding="utf-8")),
        "MFCC_LR_exact_config": json.loads((ROOT / "40_bh_models/mfcc_lr/AQ2D0BH_MFCC_LR_config.json").read_text(encoding="utf-8")),
        "anti_fishing_statement": "NO ALTERNATIVE MODEL OR REPRESENTATION WILL BE SUBSTITUTED.",
    }
    write_json(ROOT / "03_manifests/AQ2D0BH_DESIGN_FREEZE.json", design)
    arm = {
        "phase": "AQ-2D.0-BH", "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_arm_frozen": "YES", "RF_model_sha256": rf_sha, "RF_validation_metrics": rf_val,
        "RF_validation_competent": rf_val["RF_VALIDATION_COMPETENT"], "RF_formal_breadth_authorized": "YES" if rf_val["RF_VALIDATION_COMPETENT"] else "NO",
        "MFCC_spec_sha256": sha256_file(ROOT / "41_bh_representation/mfcc_spec/AQ2D0BH_MFCC_STATISTICS_V1.json"),
        "MFCC_extractor_sha256": sha256_file(ROOT / "41_bh_representation/mfcc_extractor/aq2d0bh_mfcc_extractor.py"),
        "MFCC_model_sha256": mfcc_state["numeric_sha256"], "MFCC_validation_metrics": mfcc_state["validation"],
        "MFCC_validation_competent": mfcc_state["validation"]["MFCC_VALIDATION_COMPETENT"], "MFCC_formal_breadth_authorized": mfcc_state["authorized"],
        "speech_scientific_prediction_count": 0,
    }
    write_json(ROOT / "03_manifests/AQ2D0BH_ARM_FREEZE.json", arm)
    return arm


def generate_derivatives(parents):
    rows = []
    source_samples = {p["speech_parent_uid"]: wav_read(p["canonical_pcm_path"])[0] for p in parents}
    for p in sorted(parents, key=lambda r: r["speech_parent_uid"]):
        for codec in CLASS_ORDER:
            for rate in RATES:
                ci = codec_info(codec, rate)
                uid_source = f"AQ2D0BH_DERIV_V1|{p['speech_parent_uid']}|{codec}|{rate}|{ci['core']}|{EXPECTED[str(ENCODER)]}"
                duid = sha256_text(uid_source)[:24]
                enc = ROOT / "42_bh_derivatives/encoded" / f"{duid}.{ci['ext']}"
                dec = ROOT / "42_bh_derivatives/decoded_raw" / f"{duid}.wav"
                ali = ROOT / "42_bh_derivatives/aligned_pcm" / f"{duid}.wav"
                enc_r = run_cmd([str(ENCODER), "-y", "-hide_banner", "-i", p["canonical_pcm_path"], "-map_metadata", "-1", "-vn", "-sn", "-dn"] + ci["args"] + [str(enc)], timeout=180)
                enc_ok = enc_r["return_code"] == 0 and enc.exists()
                info, probe_r = ffprobe_audio(enc) if enc_ok else (None, {"command": "", "return_code": 1, "stdout": "", "stderr": "encode failed"})
                probe_ok = info is not None and info["codec_name"] == ci["probe"]
                dec_r = run_cmd([str(DECODER), "-y", "-hide_banner", "-i", str(enc), "-map_metadata", "-1", "-vn", "-sn", "-dn", "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(dec)], timeout=180) if probe_ok else {"command": "", "return_code": 1, "stdout": "", "stderr": "probe failed"}
                dec_ok = dec_r["return_code"] == 0 and dec.exists()
                align = {"lag": "", "corr": "", "post_count": "", "leading_crop": "", "success": False}
                aligned_sha = ""
                if dec_ok:
                    dec_samp, dec_info = wav_read(dec)
                    if dec_info["sample_rate"] == 48000 and dec_info["channels"] == 1 and dec_info["sample_width"] == 2:
                        align = best_alignment(source_samples[p["speech_parent_uid"]], dec_samp)
                        if align["success"] and align["post_count"] == SEGMENT_SAMPLES:
                            start = int(align["leading_crop"])
                            wav_write(ali, dec_samp[start:start + SEGMENT_SAMPLES])
                            aligned_sha = raw_pcm_sha256(ali)
                ok = enc_ok and probe_ok and dec_ok and ali.exists() and align["success"] and align["post_count"] == SEGMENT_SAMPLES
                rows.append({
                    "derivative_uid": duid, "speech_parent_uid": p["speech_parent_uid"], "speaker_id": p["speaker_id"],
                    "codec_family": codec, "encoder_name": ci["encoder"], "encoder_core": ci["core"], "nominal_rate_kbps": rate,
                    "achieved_rate_kbps": (8 * enc.stat().st_size / SEGMENT_SECONDS / 1000.0) if enc.exists() else "",
                    "encoded_sha256": sha256_file(enc) if enc.exists() else "", "aligned_pcm_sha256": aligned_sha,
                    "alignment_lag_samples": align["lag"], "alignment_correlation": align["corr"], "aligned_sample_count": SEGMENT_SAMPLES if ok else "",
                    "encoded_size_bytes": enc.stat().st_size if enc.exists() else "", "encoded_path": str(enc) if enc.exists() else "",
                    "decoded_raw_path": str(dec) if dec.exists() else "", "aligned_pcm_path": str(ali) if ali.exists() else "",
                    "derivative_id_source_string": uid_source, "encode_return_code": enc_r["return_code"], "probe_return_code": probe_r["return_code"],
                    "decode_return_code": dec_r["return_code"], "probe_codec_name": info["codec_name"] if info else "", "status": "PASS" if ok else "FAIL",
                    "encode_command": enc_r["command"], "probe_command": probe_r["command"], "decode_command": dec_r["command"],
                })
    write_csv(ROOT / "42_bh_derivatives/AQ2D0BH_speech_derivative_manifest.csv", rows)
    files = sorted(list((ROOT / "42_bh_derivatives/encoded").glob("*")) + list((ROOT / "42_bh_derivatives/decoded_raw").glob("*")) + list((ROOT / "42_bh_derivatives/aligned_pcm").glob("*")), key=lambda x: x.name)
    (ROOT / "42_bh_derivatives/AQ2D0BH_derivative_file_sha256s.txt").write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in files) + "\n", encoding="utf-8")
    ach = []
    for codec in CLASS_ORDER:
        for rate in RATES:
            vals = [float(r["achieved_rate_kbps"]) for r in rows if r["codec_family"] == codec and r["nominal_rate_kbps"] == rate]
            ach.append({"codec": codec, "rate": rate, "n": len(vals), "mean_achieved_rate_kbps": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))})
    write_csv(ROOT / "42_bh_derivatives/audit/AQ2D0BH_achieved_rate_summary.csv", ach)
    by_pcm = defaultdict(list)
    for r in rows:
        by_pcm[r["aligned_pcm_sha256"]].append(r["derivative_uid"])
    collisions = [{"aligned_pcm_sha256": k, "count": len(v), "derivative_uids": ";".join(v)} for k, v in by_pcm.items() if k and len(v) > 1]
    write_csv(ROOT / "42_bh_derivatives/audit/AQ2D0BH_collision_audit.csv", collisions or [{"aligned_pcm_sha256": "", "count": 0, "derivative_uids": ""}])
    if len(rows) != 480 or sum(r["status"] == "PASS" for r in rows) != 480:
        fail("BLOCKED_EXECUTION", "SPEECH_DERIVATIVE_MATRIX_INCOMPLETE", {"rows": len(rows), "pass": sum(r["status"] == "PASS" for r in rows)})
    return rows


def extract_speech_features(derivs, arm):
    rep_spec, rep_mod = load_rep_a()
    repa_rows = []
    for r in derivs:
        arr = rep_a_from_wav(Path(r["aligned_pcm_path"]), rep_spec, rep_mod)
        npy = ROOT / "41_bh_representation/repA_speech_features" / f"{r['derivative_uid']}.npy"
        np.save(npy, arr, allow_pickle=False)
        repa_rows.append({"derivative_uid": r["derivative_uid"], "array_sha256": canonical_array_hash(arr, "STFT_LOGPOWER_STATS_V1"), "npy_file_sha256": sha256_file(npy), "shape": "256", "dtype": "float64", "finite_check": True, "npy_path": str(npy)})
    write_csv(ROOT / "41_bh_representation/AQ2D0BH_speech_repA_feature_manifest.csv", repa_rows)
    mfcc_rows = []
    if arm["MFCC_formal_breadth_authorized"] == "YES":
        for r in derivs:
            arr = extract_mfcc(Path(r["aligned_pcm_path"]))
            npy = ROOT / "41_bh_representation/speech_features" / f"{r['derivative_uid']}.npy"
            np.save(npy, arr, allow_pickle=False)
            mfcc_rows.append({"derivative_uid": r["derivative_uid"], "array_sha256": canonical_array_hash(arr, "MFCC_STATISTICS_V1"), "npy_file_sha256": sha256_file(npy), "shape": "40", "dtype": "float64", "finite_check": True, "npy_path": str(npy)})
        write_csv(ROOT / "41_bh_representation/AQ2D0BH_speech_MFCC_feature_manifest.csv", mfcc_rows)
    return dict(zip([r["derivative_uid"] for r in repa_rows], [r["npy_path"] for r in repa_rows])), dict(zip([r["derivative_uid"] for r in mfcc_rows], [r["npy_path"] for r in mfcc_rows]))


def prepredict_freeze(parents, derivs, repa_map, mfcc_map, arm):
    freeze = {
        "phase": "AQ-2D.0-BH", "created_utc": datetime.now(timezone.utc).isoformat(), "speech_parent_count": 40,
        "expected_derivatives": 480, "actual_derivatives": len(derivs), "encode_success": sum(r["encode_return_code"] == 0 for r in derivs),
        "probe_success": sum(r["probe_return_code"] == 0 for r in derivs), "decode_success": sum(r["decode_return_code"] == 0 for r in derivs),
        "alignment_success": sum(r["status"] == "PASS" for r in derivs), "RepA_feature_success": len(repa_map),
        "MFCC_feature_success": len(mfcc_map) if arm["MFCC_formal_breadth_authorized"] == "YES" else "NOT_APPLICABLE",
        "primary_model_identity": "Frozen AQ2B1 STFT_LOGPOWER_STATS_V1 + StandardScaler + multinomial LogisticRegression",
        "RF_arm_eligibility": arm["RF_formal_breadth_authorized"], "MFCC_arm_eligibility": arm["MFCC_formal_breadth_authorized"],
        "speech_scientific_prediction_count": 0, "scientific_speech_metrics_computed": "NO",
        "artifact_hashes": {
            "speech_parent_manifest": sha256_file(ROOT / "39_bh_speech/AQ2D0BH_speech_parent_manifest.csv"),
            "speech_derivative_manifest": sha256_file(ROOT / "42_bh_derivatives/AQ2D0BH_speech_derivative_manifest.csv"),
            "repA_speech_feature_manifest": sha256_file(ROOT / "41_bh_representation/AQ2D0BH_speech_repA_feature_manifest.csv"),
            "mfcc_speech_feature_manifest": sha256_file(ROOT / "41_bh_representation/AQ2D0BH_speech_MFCC_feature_manifest.csv") if mfcc_map else "",
            "primary_model": EXPECTED["18_model_b1/AQ2B1_model_numeric.npz"],
            "RF_model": arm["RF_model_sha256"] if arm["RF_formal_breadth_authorized"] == "YES" else "",
            "MFCC_model": arm["MFCC_model_sha256"] if arm["MFCC_formal_breadth_authorized"] == "YES" else "",
        },
    }
    write_json(ROOT / "03_manifests/AQ2D0BH_PREPREDICTION_FREEZE.json", freeze)


def softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def distances(z, centroids):
    vals = {c: float(np.linalg.norm(z - centroids[c])) for c in CLASS_ORDER}
    true = None
    return vals


def score_linear_arm(derivs, feature_map, npz_path, centroid_path, out_path, prefix=""):
    model = np.load(npz_path, allow_pickle=False)
    cent = np.load(centroid_path, allow_pickle=False)
    centroids = {c: cent[f"centroid_{c}"] for c in CLASS_ORDER}
    rows = []
    for r in derivs:
        x = np.load(feature_map[r["derivative_uid"]], allow_pickle=False)
        z = (x - model["scaler_mean"]) / model["scaler_scale"]
        logits = z @ model["model_coef"].T + model["model_intercept"]
        probs = softmax(logits)
        pred = CLASS_ORDER[int(np.argmax(logits))]
        ti = CLASS_ORDER.index(r["codec_family"])
        d = distances(z, centroids)
        dtrue = d[r["codec_family"]]
        wrong_min = min(v for k, v in d.items() if k != r["codec_family"])
        rank = 1 + sum(1 for k, v in d.items() if k != r["codec_family"] and v < dtrue)
        rows.append({"derivative_uid": r["derivative_uid"], "speech_parent_uid": r["speech_parent_uid"], "speaker_id": r["speaker_id"], "true_codec": r["codec_family"], "nominal_rate_kbps": r["nominal_rate_kbps"], "predicted_codec": pred, "prob_AAC": float(probs[0]), "prob_MP3": float(probs[1]), "prob_Opus": float(probs[2]), "true_margin": float(logits[ti] - max(logits[j] for j in range(3) if j != ti)), "distance_centroid_AAC": d["AAC"], "distance_centroid_MP3": d["MP3"], "distance_centroid_Opus": d["Opus"], "wrong_centroid_closer": bool(wrong_min < dtrue), "true_centroid_rank": rank, "achieved_rate_kbps": r["achieved_rate_kbps"]})
    write_csv(out_path, rows)
    return rows


def score_rf_arm(rf, derivs, feature_map, out_path):
    model = np.load(ROOT / "18_model_b1/AQ2B1_model_numeric.npz", allow_pickle=False)
    cent = np.load(ROOT / "29_reference_c1b/centroids/AQ2C1B_B1_train_centroids.npz", allow_pickle=False)
    centroids = {c: cent[f"centroid_{c}"] for c in CLASS_ORDER}
    rows = []
    for r in derivs:
        x = np.load(feature_map[r["derivative_uid"]], allow_pickle=False)
        z = (x - model["scaler_mean"]) / model["scaler_scale"]
        probs_by_model = dict(zip(rf.classes_, rf.predict_proba(z.reshape(1, -1))[0]))
        probs = np.array([probs_by_model[c] for c in CLASS_ORDER], dtype=np.float64)
        pred = CLASS_ORDER[int(np.argmax(probs))]
        ti = CLASS_ORDER.index(r["codec_family"])
        d = distances(z, centroids)
        dtrue = d[r["codec_family"]]
        wrong_min = min(v for k, v in d.items() if k != r["codec_family"])
        rank = 1 + sum(1 for k, v in d.items() if k != r["codec_family"] and v < dtrue)
        rows.append({"derivative_uid": r["derivative_uid"], "speech_parent_uid": r["speech_parent_uid"], "speaker_id": r["speaker_id"], "true_codec": r["codec_family"], "nominal_rate_kbps": r["nominal_rate_kbps"], "predicted_codec": pred, "prob_AAC": float(probs[0]), "prob_MP3": float(probs[1]), "prob_Opus": float(probs[2]), "true_prob_margin": float(probs[ti] - max(probs[j] for j in range(3) if j != ti)), "distance_centroid_AAC": d["AAC"], "distance_centroid_MP3": d["MP3"], "distance_centroid_Opus": d["Opus"], "wrong_centroid_closer": bool(wrong_min < dtrue), "true_centroid_rank": rank, "achieved_rate_kbps": r["achieved_rate_kbps"]})
    write_csv(out_path, rows)
    return rows


def boundary_summary(rows, arm_name, margin_key, seeds):
    endpoint = endpoint_metrics([{**r, "codec_family": r["true_codec"]} for r in rows], "predicted_codec")
    endpoint[f"{arm_name}_SPEECH_ENDPOINT_PASS"] = endpoint_pass(endpoint)
    endpoint[f"{arm_name}_SPEECH_ENDPOINT_SEVERE_FAIL"] = endpoint_severe(endpoint)
    endpoint_floor = min(endpoint["AAC_RECALL_32"], endpoint["AAC_RECALL_128"])
    events = {}
    boot_rows_all = []
    for rate in [80, 96]:
        aac = [r for r in rows if r["true_codec"] == "AAC" and int(r["nominal_rate_kbps"]) == rate]
        recall = sum(r["predicted_codec"] == "AAC" for r in aac) / len(aac)
        drop = endpoint_floor - recall
        margins = [float(r[margin_key]) for r in aac]
        wrong_frac = float(np.mean([str(r["wrong_centroid_closer"]).lower() == "true" or r["wrong_centroid_closer"] is True for r in aac]))
        wrong_preds = [r["predicted_codec"] for r in aac if r["predicted_codec"] != "AAC"]
        if wrong_preds:
            dom, dom_n = Counter(wrong_preds).most_common(1)[0]
            dom_share = dom_n / len(wrong_preds)
        else:
            dom, dom_share = "", 0.0
        structural = recall <= 0.50 and drop >= 0.25 and float(np.median(margins)) < 0 and wrong_frac >= 0.60 and dom_share >= 0.60
        boot = bootstrap_drop(rows, rate, seeds[rate])
        boot_rows_all.extend([{**b, "arm": arm_name, "rate": rate} for b in boot["rows"]])
        events[str(rate)] = {"AAC_RECALL": recall, "AAC_RECALL_DROP": drop, "AAC_MEDIAN_TRUE_MARGIN": float(np.median(margins)), "AAC_WRONG_CENTROID_FRACTION": wrong_frac, "AAC_DOMINANT_ERROR_TARGET": dom, "AAC_DOMINANT_ERROR_SHARE": dom_share, "STRUCTURAL_EVENT": bool(structural), "BOOTSTRAP_LOW": boot["low"], "BOOTSTRAP_MEDIAN": boot["median"], "BOOTSTRAP_HIGH": boot["high"], "PAIRED_DROP_STABLE": boot["stable"], f"FULL_{arm_name}_SPEECH_EVENT": bool(structural and boot["stable"])}
    write_csv(ROOT / "43_bh_results/bootstrap" / f"AQ2D0BH_{arm_name}_bootstrap.csv", boot_rows_all)
    return {"endpoint": endpoint, "events_80_96": events}


def bootstrap_drop(rows, rate, seed):
    aac = [r for r in rows if r["true_codec"] == "AAC"]
    by_speaker = defaultdict(dict)
    for r in aac:
        by_speaker[r["speaker_id"]][int(r["nominal_rate_kbps"])] = r["predicted_codec"] == "AAC"
    speakers = sorted(by_speaker)
    drops = []
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        sample = rng.choice(speakers, size=len(speakers), replace=True)
        vals = []
        for sp in sample:
            ep = 0.5 * (float(by_speaker[sp][32]) + float(by_speaker[sp][128]))
            vals.append(ep - float(by_speaker[sp][rate]))
        drops.append(float(np.mean(vals)))
    low, med, high = np.percentile(drops, [2.5, 50, 97.5])
    return {"low": float(low), "median": float(med), "high": float(high), "stable": bool(low > 0), "rows": [{"replicate": i, "drop": v} for i, v in enumerate(drops)]}


def classify_and_freeze(record, parents, derivs, arm, summaries, created_hashes):
    primary = summaries["PRIMARY"]
    rf = summaries.get("RF")
    mfcc = summaries.get("MFCC")
    content_strong = primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"] and any(v["FULL_PRIMARY_SPEECH_EVENT"] for v in primary["events_80_96"].values())
    content_dep = primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"] and not content_strong
    content_unresolved = not primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"]
    rf_strong = bool(rf and arm["RF_formal_breadth_authorized"] == "YES" and rf["endpoint"]["RF_SPEECH_ENDPOINT_PASS"] and primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"] and any(primary["events_80_96"][str(r)]["FULL_PRIMARY_SPEECH_EVENT"] and rf["events_80_96"][str(r)]["FULL_RF_SPEECH_EVENT"] for r in [80, 96]))
    mfcc_strong = bool(mfcc and arm["MFCC_formal_breadth_authorized"] == "YES" and mfcc["endpoint"]["MFCC_SPEECH_ENDPOINT_PASS"] and primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"] and any(primary["events_80_96"][str(r)]["FULL_PRIMARY_SPEECH_EVENT"] and mfcc["events_80_96"][str(r)]["FULL_MFCC_SPEECH_EVENT"] for r in [80, 96]))
    model_dep = bool(rf and arm["RF_formal_breadth_authorized"] == "YES" and rf["endpoint"]["RF_SPEECH_ENDPOINT_PASS"] and primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"] and any(primary["events_80_96"][str(r)]["FULL_PRIMARY_SPEECH_EVENT"] for r in [80, 96]) and not any(rf["events_80_96"][str(r)]["FULL_RF_SPEECH_EVENT"] for r in [80, 96]))
    rep_dep = bool(mfcc and arm["MFCC_formal_breadth_authorized"] == "YES" and mfcc["endpoint"]["MFCC_SPEECH_ENDPOINT_PASS"] and primary["endpoint"]["PRIMARY_SPEECH_ENDPOINT_PASS"] and any(primary["events_80_96"][str(r)]["FULL_PRIMARY_SPEECH_EVENT"] for r in [80, 96]) and not any(mfcc["events_80_96"][str(r)]["FULL_MFCC_SPEECH_EVENT"] for r in [80, 96]))
    if content_strong and rf_strong and mfcc_strong:
        final = "PASS_FULL_Q1_BREADTH_HARDENING"
    elif content_strong and (rf_strong + mfcc_strong == 1):
        final = "PASS_PARTIAL_Q1_BREADTH_HARDENING"
    elif content_strong and not rf_strong and not mfcc_strong:
        final = "PASS_CONTENT_BREADTH_ONLY"
    elif model_dep or rep_dep or content_dep:
        final = "VALID_BREADTH_LIMITATION_SIGNAL"
    else:
        final = "BREADTH_PROBES_INADEQUATE"
    scorecard = {
        "CONTENT_BREADTH": "STRONG" if content_strong else ("MODERATE / UNRESOLVED: endpoint incompetent on external speech domain" if content_unresolved else "MODERATE / UNRESOLVED: endpoint competent but no full speech event"),
        "MODEL_BREADTH": "STRONG" if rf_strong else "LIMITED / UNRESOLVED: RF validation or speech/event prerequisite not satisfied",
        "REPRESENTATION_BREADTH": "STRONG" if mfcc_strong else "LIMITED / UNRESOLVED: MFCC validation or speech/event prerequisite not satisfied",
    }
    write_json(ROOT / "43_bh_results/gate/AQ2D0BH_Q1_breadth_scorecard.json", scorecard)
    gate = {
        "final_classification": final, "CONTENT_BREADTH_STRONG": content_strong, "MODEL_BREADTH_STRONG": rf_strong,
        "REPRESENTATION_BREADTH_STRONG": mfcc_strong, "CONTENT_DOMAIN_DEPENDENCE_SIGNAL": content_dep,
        "MODEL_DEPENDENCE_SIGNAL": model_dep, "REPRESENTATION_DEPENDENCE_SIGNAL": rep_dep,
        "CONTENT_BREADTH_UNRESOLVED_EXTERNAL_DOMAIN": content_unresolved,
        "RF_FORMAL_BREADTH_AUTHORIZED": arm["RF_formal_breadth_authorized"],
        "MFCC_FORMAL_BREADTH_AUTHORIZED": arm["MFCC_formal_breadth_authorized"],
        "final_Q1_breadth_scorecard": scorecard,
    }
    write_json(ROOT / "43_bh_results/gate/AQ2D0BH_breadth_gate.json", gate)
    freeze = {
        "phase": "AQ-2D.0-BH", "specification_version": "v1.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_classification": final, "external_corpus": "LibriSpeech test-clean", "archive_md5": record["observed_md5"],
        "archive_sha256": record["archive_sha256"], "license": "CC BY 4.0", "speech_parent_count": 40, "unique_speaker_count": 40,
        "speech_parent_manifest_sha256": sha256_file(ROOT / "39_bh_speech/AQ2D0BH_speech_parent_manifest.csv"),
        "codec_set": CLASS_ORDER, "rate_grid": RATES, "expected_derivatives": 480, "actual_derivatives": len(derivs),
        "encode_success": sum(r["encode_return_code"] == 0 for r in derivs), "probe_success": sum(r["probe_return_code"] == 0 for r in derivs),
        "decode_success": sum(r["decode_return_code"] == 0 for r in derivs), "alignment_success": sum(r["status"] == "PASS" for r in derivs),
        "primary_model_identity": "AQ2B1 frozen Representation-A LogisticRegression",
        "primary_speech_endpoint_metrics": primary["endpoint"], "primary_full_events_80_96": primary["events_80_96"],
        "RF_validation_metrics": arm["RF_validation_metrics"], "RF_validation_competent": arm["RF_validation_competent"],
        "RF_speech_endpoint_metrics": rf["endpoint"] if rf else "NOT_AUTHORIZED", "RF_full_events_80_96": rf["events_80_96"] if rf else "NOT_AUTHORIZED",
        "MFCC_representation_identity": "MFCC_STATISTICS_V1", "MFCC_validation_metrics": arm["MFCC_validation_metrics"],
        "MFCC_validation_competent": arm["MFCC_validation_competent"], "MFCC_speech_endpoint_metrics": mfcc["endpoint"] if mfcc else "NOT_AUTHORIZED",
        "MFCC_full_events_80_96": mfcc["events_80_96"] if mfcc else "NOT_AUTHORIZED",
        "CONTENT_BREADTH_STRONG": content_strong, "MODEL_BREADTH_STRONG": rf_strong, "REPRESENTATION_BREADTH_STRONG": mfcc_strong,
        "CONTENT_DOMAIN_DEPENDENCE_SIGNAL": content_dep, "MODEL_DEPENDENCE_SIGNAL": model_dep, "REPRESENTATION_DEPENDENCE_SIGNAL": rep_dep,
        "final_Q1_breadth_scorecard": scorecard, "C1B_result_changed": "NO", "C2_result_changed": "NO", "C2_DX_result_changed": "NO",
        "BH_R2_authorized": "NO", "new_training_performed": {"RF": "YES", "MFCC_LR": "YES"}, "B1_model_refit_performed": "NO",
        "C1B_data_used_for_training": "NO", "C2_data_used_for_training": "NO", "speech_data_used_for_training": "NO",
        "vorbis_used": "NO", "representation_B_used": "NO", "representation_C_rescue_used": "NO",
        "AQ2A_v1_frozen_artifacts_modified": "NO", "AQ2A_R1_frozen_artifacts_modified": "NO", "AQ2A_R2_frozen_artifacts_modified": "NO",
        "AQ2B0_frozen_artifacts_modified": "NO", "AQ2B1_frozen_artifacts_modified": "NO", "AQ2B1_DX_frozen_artifacts_modified": "NO",
        "AQ2C1A_frozen_artifacts_modified": "NO", "AQ2C1B_frozen_artifacts_modified": "NO", "AQ2C2_frozen_artifacts_modified": "NO",
        "AQ2C2_DX_frozen_artifacts_modified": "NO", "existing_MM_artifacts_modified": "NO", "artifact_hashes": created_hashes,
    }
    write_json(ROOT / "03_manifests/AQ2D0BH_FREEZE.json", freeze)
    write_report(freeze, gate, arm)
    return freeze, gate


def write_report(freeze, gate, arm):
    items = [
        ("1. Final BH classification", freeze["final_classification"]),
        ("2. Scientific-scope statement", "One-shot manuscript breadth hardening only; no universal or model-independent claim is authorized beyond passed axes."),
        ("3. Confirmation that BH is not a C2 rescue", "BH is separately governed and does not modify AQ-2C.2 or AQ-2C.2-DX."),
        ("4. Predecessor integrity reconciliation", "PASS for required C1B, C2, C2-DX, primary model, RepA, Python, and codec hashes."),
        ("5. LibriSpeech source identity", "OpenSLR SLR12 test-clean."),
        ("6. License", "CC BY 4.0."),
        ("7. Archive checksum verification", f"MD5={freeze['archive_md5']}; SHA256={freeze['archive_sha256']}."),
        ("8. Extracted utterance count", "2620 expected; see inventory artifact."),
        ("9. Unique speaker count", str(freeze["unique_speaker_count"])),
        ("10. Deterministic selection method", "SHA256(AQ2D0BH_SPEECH_SELECTION_V1|speaker_id|utterance_id), first eligible per speaker."),
        ("11. Final 40-parent / 40-speaker cohort", f"{freeze['speech_parent_count']} / {freeze['unique_speaker_count']}."),
        ("12. Canonicalization integrity", "48 kHz mono PCM16, 2.0 s, center crop, 96000 samples, 40 unique PCM hashes."),
        ("13. Codec/rate matrix", "AAC, MP3, Opus at 32, 80, 96, 128 kbps."),
        ("14. Expected and actual derivatives", f"{freeze['expected_derivatives']} / {freeze['actual_derivatives']}."),
        ("15. Encode/probe/decode/alignment result", f"{freeze['encode_success']} / {freeze['probe_success']} / {freeze['decode_success']} / {freeze['alignment_success']}."),
        ("16. Primary A+LR identity", freeze["primary_model_identity"]),
        ("17. RF exact configuration", json.dumps(RF_CONFIG, sort_keys=True)),
        ("18. RF B1-validation competence result", json.dumps(arm["RF_validation_metrics"], sort_keys=True)),
        ("19. MFCC representation motivation", "Prior-art-grounded cepstral representation, fixed one-shot alternative to STFT log-power statistics."),
        ("20. Exact MFCC specification", "See 41_bh_representation/mfcc_spec/AQ2D0BH_MFCC_STATISTICS_V1.json."),
        ("21. MFCC determinism result", "PASS; RUN1/RUN2/RUN3 exact equality for B1 TRAIN."),
        ("22. MFCC-LR B1-validation competence result", json.dumps(arm["MFCC_validation_metrics"], sort_keys=True)),
        ("23. ARM FREEZE SHA-256", sha256_file(ROOT / "03_manifests/AQ2D0BH_ARM_FREEZE.json")),
        ("24. PREPREDICTION FREEZE SHA-256", sha256_file(ROOT / "03_manifests/AQ2D0BH_PREPREDICTION_FREEZE.json")),
        ("25. Proof speech prediction count was zero at freeze", "AQ2D0BH_PREPREDICTION_FREEZE.json records speech_scientific_prediction_count = 0."),
        ("26. Primary speech endpoint BA", freeze["primary_speech_endpoint_metrics"]["BA"]),
        ("27. Primary AAC 32/128 recalls", f"{freeze['primary_speech_endpoint_metrics']['AAC_RECALL_32']} / {freeze['primary_speech_endpoint_metrics']['AAC_RECALL_128']}"),
        ("28. Primary AAC 80 result", json.dumps(freeze["primary_full_events_80_96"]["80"], sort_keys=True)),
        ("29. Primary AAC 96 result", json.dumps(freeze["primary_full_events_80_96"]["96"], sort_keys=True)),
        ("30. Primary bootstrap results", "See 43_bh_results/bootstrap/AQ2D0BH_PRIMARY_bootstrap.csv."),
        ("31. CONTENT_BREADTH_STRONG status", freeze["CONTENT_BREADTH_STRONG"]),
        ("32. RF speech endpoint result if authorized", json.dumps(freeze["RF_speech_endpoint_metrics"], sort_keys=True) if isinstance(freeze["RF_speech_endpoint_metrics"], dict) else freeze["RF_speech_endpoint_metrics"]),
        ("33. RF 80/96 events if authorized", json.dumps(freeze["RF_full_events_80_96"], sort_keys=True) if isinstance(freeze["RF_full_events_80_96"], dict) else freeze["RF_full_events_80_96"]),
        ("34. MODEL_BREADTH_STRONG status", freeze["MODEL_BREADTH_STRONG"]),
        ("35. MFCC speech endpoint result if authorized", json.dumps(freeze["MFCC_speech_endpoint_metrics"], sort_keys=True) if isinstance(freeze["MFCC_speech_endpoint_metrics"], dict) else freeze["MFCC_speech_endpoint_metrics"]),
        ("36. MFCC 80/96 events if authorized", json.dumps(freeze["MFCC_full_events_80_96"], sort_keys=True) if isinstance(freeze["MFCC_full_events_80_96"], dict) else freeze["MFCC_full_events_80_96"]),
        ("37. REPRESENTATION_BREADTH_STRONG status", freeze["REPRESENTATION_BREADTH_STRONG"]),
        ("38. Any content dependence signal", freeze["CONTENT_DOMAIN_DEPENDENCE_SIGNAL"]),
        ("39. Any model dependence signal", freeze["MODEL_DEPENDENCE_SIGNAL"]),
        ("40. Any representation dependence signal", freeze["REPRESENTATION_DEPENDENCE_SIGNAL"]),
        ("41. Final Q1 breadth scorecard", json.dumps(freeze["final_Q1_breadth_scorecard"], sort_keys=True)),
        ("42. Implication for manuscript claim boundary", "Use only axes that passed; unresolved axes remain limited."),
        ("43. Explicit no-rescue statement", "No C2 rescue, no Representation C rescue, no second classifier, no second representation."),
        ("44. All artifacts", "See AQ2D0BH_SHA256SUMS.txt."),
        ("45. Major hashes", json.dumps({"FREEZE": sha256_file(ROOT / "03_manifests/AQ2D0BH_FREEZE.json"), "DESIGN": sha256_file(ROOT / "03_manifests/AQ2D0BH_DESIGN_FREEZE.json"), "ARM": sha256_file(ROOT / "03_manifests/AQ2D0BH_ARM_FREEZE.json"), "PREPREDICTION": sha256_file(ROOT / "03_manifests/AQ2D0BH_PREPREDICTION_FREEZE.json")}, sort_keys=True)),
        ("46. Remaining limitations", "No BH-R2; no additional speech corpus, model, or representation attempted."),
        ("Explicit statements", "\n".join([
            "AQ-2B.1 RESULT REMAINS = STOP_MFEA1_TRACE_NOT_SUPPORTED",
            "AQ-2C.1B RESULT REMAINS = PASS_NEW_RATE_BOUNDARY_CONFIRMED",
            "AQ-2C.2 RESULT REMAINS = TRIANGULATION_PROBE_INADEQUATE",
            "AQ-2C.2-DX RESULT REMAINS = DX_INCONCLUSIVE",
            "BH-R2 AUTHORIZED = NO",
            "NEW RF TRAINING PERFORMED = YES",
            "NEW MFCC-LR TRAINING PERFORMED = YES",
            "B1 MODEL REFIT PERFORMED = NO",
            "C1B DATA USED FOR TRAINING = NO",
            "C2 DATA USED FOR TRAINING = NO",
            "SPEECH DATA USED FOR TRAINING = NO",
            "ALTERNATIVE MODEL SWEEP PERFORMED = NO",
            "ALTERNATIVE REPRESENTATION SWEEP PERFORMED = NO",
            "VORBIS USED = NO",
            "REPRESENTATION B USED = NO",
            "REPRESENTATION C RESCUE USED = NO",
            "ALL PREDECESSOR ARTIFACTS MODIFIED = NO",
            "MM ARTIFACTS MODIFIED = NO",
        ])),
    ]
    text = "\n\n".join(f"## {title}\n{body}" for title, body in items) + "\n"
    (ROOT / "44_bh_reports/AQ2D0BH_EXECUTION_REPORT.md").write_text(text, encoding="utf-8")


def write_ledger():
    files = [ROOT / "03_manifests/AQ2D0BH_DESIGN_FREEZE.json", ROOT / "03_manifests/AQ2D0BH_ARM_FREEZE.json", ROOT / "03_manifests/AQ2D0BH_PREPREDICTION_FREEZE.json", ROOT / "03_manifests/AQ2D0BH_FREEZE.json"]
    for d in ["39_bh_speech", "40_bh_models", "41_bh_representation", "42_bh_derivatives", "43_bh_results", "44_bh_reports", "logs_bh"]:
        files.extend(sorted((ROOT / d).rglob("*")))
    files = [p for p in files if p.is_file() and p.name != "AQ2D0BH_SHA256SUMS.txt"]
    lines = [f"{sha256_file(p)}  {p.relative_to(ROOT).as_posix()}" for p in sorted(set(files))]
    (ROOT / "AQ2D0BH_SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    mkdirs()
    verify_predecessors()
    record = acquire_librispeech()
    test_clean = extract_archive()
    inventory = inventory_librispeech(test_clean)
    parents = select_speech_parents(inventory)
    _, fmap, train, val = load_b1()
    rf, rf_sha = fit_rf(train, fmap)
    rf_val = validate_rf(rf, val, fmap)
    mfcc_state = mfcc_determinism_and_fit(train, val)
    arm = design_and_arm_freezes(record, parents, rf_sha, rf_val, mfcc_state)
    derivs = generate_derivatives(parents)
    repa_map, mfcc_map = extract_speech_features(derivs, arm)
    prepredict_freeze(parents, derivs, repa_map, mfcc_map, arm)
    summaries = {}
    primary_rows = score_linear_arm(derivs, repa_map, ROOT / "18_model_b1/AQ2B1_model_numeric.npz", ROOT / "29_reference_c1b/centroids/AQ2C1B_B1_train_centroids.npz", ROOT / "43_bh_results/primary/AQ2D0BH_primary_speech_predictions.csv")
    summaries["PRIMARY"] = boundary_summary(primary_rows, "PRIMARY", "true_margin", {80: 20261180, 96: 20261196})
    write_json(ROOT / "43_bh_results/primary/AQ2D0BH_primary_summary.json", summaries["PRIMARY"])
    if arm["RF_formal_breadth_authorized"] == "YES":
        rf_rows = score_rf_arm(rf, derivs, repa_map, ROOT / "43_bh_results/rf/AQ2D0BH_RF_speech_predictions.csv")
        summaries["RF"] = boundary_summary(rf_rows, "RF", "true_prob_margin", {80: 20261280, 96: 20261296})
        write_json(ROOT / "43_bh_results/rf/AQ2D0BH_RF_summary.json", summaries["RF"])
    else:
        write_json(ROOT / "43_bh_results/rf/AQ2D0BH_RF_summary.json", {"status": "NOT_AUTHORIZED", "RF_validation_metrics": rf_val})
    if arm["MFCC_formal_breadth_authorized"] == "YES":
        mfcc_npz = ROOT / "40_bh_models/mfcc_lr/AQ2D0BH_MFCC_LR_numeric.npz"
        mfcc_cent = ROOT / "40_bh_models/mfcc_lr/AQ2D0BH_MFCC_train_centroids.npz"
        mfcc_rows = score_linear_arm(derivs, mfcc_map, mfcc_npz, mfcc_cent, ROOT / "43_bh_results/mfcc_lr/AQ2D0BH_MFCC_LR_speech_predictions.csv")
        summaries["MFCC"] = boundary_summary(mfcc_rows, "MFCC", "true_margin", {80: 20261380, 96: 20261396})
        write_json(ROOT / "43_bh_results/mfcc_lr/AQ2D0BH_MFCC_summary.json", summaries["MFCC"])
    else:
        write_json(ROOT / "43_bh_results/mfcc_lr/AQ2D0BH_MFCC_summary.json", {"status": "NOT_AUTHORIZED", "MFCC_validation_metrics": mfcc_state["validation"]})
    write_json(ROOT / "43_bh_results/comparison/AQ2D0BH_cross_arm_comparison.json", summaries)
    created_hashes = {}
    for rel in ["03_manifests/AQ2D0BH_DESIGN_FREEZE.json", "03_manifests/AQ2D0BH_ARM_FREEZE.json", "03_manifests/AQ2D0BH_PREPREDICTION_FREEZE.json"]:
        created_hashes[rel] = sha256_file(ROOT / rel)
    freeze, gate = classify_and_freeze(record, parents, derivs, arm, summaries, created_hashes)
    write_ledger()
    print(json.dumps({"final_classification": freeze["final_classification"], "CONTENT_BREADTH_STRONG": freeze["CONTENT_BREADTH_STRONG"], "MODEL_BREADTH_STRONG": freeze["MODEL_BREADTH_STRONG"], "REPRESENTATION_BREADTH_STRONG": freeze["REPRESENTATION_BREADTH_STRONG"], "ledger_sha256": sha256_file(ROOT / "AQ2D0BH_SHA256SUMS.txt")}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
