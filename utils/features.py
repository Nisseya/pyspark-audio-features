import os
import warnings
import numpy as np
import librosa

def _stats_1d(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())

def _stats_2d_rows(M, n_rows):
    M = np.asarray(M, dtype=np.float64)
    out = []
    for i in range(n_rows):
        out.extend(_stats_1d(M[i, :]))
    return out

def pipeline_features(audio_path: str) -> dict | None:
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        if y is None or len(y) < 100:
            return None

        S = np.abs(librosa.stft(y))
        y_harm = librosa.effects.harmonic(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        features = {}

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            m, s, mn, mx = _stats_1d(mfcc[i])
            features[f"mfcc_{i+1}_mean"] = m
            features[f"mfcc_{i+1}_std"]  = s
            features[f"mfcc_{i+1}_min"]  = mn
            features[f"mfcc_{i+1}_max"]  = mx

        for name, arr in [
            ("centroid",  librosa.feature.spectral_centroid(S=S, sr=sr)[0]),
            ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]),
            ("rolloff",   librosa.feature.spectral_rolloff(S=S, sr=sr)[0]),
            ("flatness",  librosa.feature.spectral_flatness(S=S)[0]),
        ]:
            m, s, mn, mx = _stats_1d(arr)
            features[f"{name}_mean"] = m
            features[f"{name}_std"]  = s
            features[f"{name}_min"]  = mn
            features[f"{name}_max"]  = mx

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(7):
            m, s, mn, mx = _stats_1d(contrast[i])
            features[f"contrast_{i+1}_mean"] = m
            features[f"contrast_{i+1}_std"]  = s
            features[f"contrast_{i+1}_min"]  = mn
            features[f"contrast_{i+1}_max"]  = mx

        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        m, s, mn, mx = _stats_1d(onset_env)
        features["onset_mean"] = m
        features["onset_std"]  = s
        features["onset_min"]  = mn
        features["onset_max"]  = mx

        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        for i in range(12):
            m, s, mn, mx = _stats_1d(chroma[i])
            features[f"chroma_{i+1}_mean"] = m
            features[f"chroma_{i+1}_std"]  = s
            features[f"chroma_{i+1}_min"]  = mn
            features[f"chroma_{i+1}_max"]  = mx

        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(6):
            m, s, mn, mx = _stats_1d(tonnetz[i])
            features[f"tonnetz_{i+1}_mean"] = m
            features[f"tonnetz_{i+1}_std"]  = s
            features[f"tonnetz_{i+1}_min"]  = mn
            features[f"tonnetz_{i+1}_max"]  = mx

        for name, arr in [
            ("rms", librosa.feature.rms(S=S)[0]),
            ("zcr", librosa.feature.zero_crossing_rate(y)[0]),
        ]:
            m, s, mn, mx = _stats_1d(arr)
            features[f"{name}_mean"] = m
            features[f"{name}_std"]  = s
            features[f"{name}_min"]  = mn
            features[f"{name}_max"]  = mx

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None