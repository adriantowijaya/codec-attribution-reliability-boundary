import hashlib
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
