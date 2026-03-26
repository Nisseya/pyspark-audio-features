import io
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import numpy as np
import pandas as pd
from google.cloud import storage
from tqdm import tqdm

BUCKET = "spark-audio-bucket"
WAV_PREFIX = "audio/wav"
TSV_BLOB = "assets/raw_30s.tsv"
MAX_WORKERS = 28

client_global = storage.Client()


# -------------------------
# STATS HELPERS
# -------------------------

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


# -------------------------
# METADATA
# -------------------------

def load_metadata() -> pd.DataFrame:
    print("Loading metadata from GCS...")
    buf = io.BytesIO()
    client_global.bucket(BUCKET).blob(TSV_BLOB).download_to_file(buf)
    buf.seek(0)
    meta = pd.read_csv(buf, sep="\t")

    meta["track_num"] = meta["TRACK_ID"].str.extract(r"track_0*([0-9]+)$").astype(int)
    meta["genres"] = meta["TAGS"].apply(
        lambda t: re.findall(r"genre---([^\t\s]+)", str(t))
    )
    meta["user_id"] = np.random.randint(0, 6, size=len(meta))

    return meta[["track_num", "TRACK_ID", "PATH", "genres", "user_id"]].rename(
        columns={"PATH": "meta_path"}
    )


# -------------------------
# FEATURE EXTRACTION
# -------------------------

def extract_features(blob_name: str) -> dict | None:
    client = storage.Client()
    buf = io.BytesIO()
    client.bucket(BUCKET).blob(blob_name).download_to_file(buf)
    buf.seek(0)

    try:
        y, sr = librosa.load(buf, sr=None, mono=True)
        if y is None or len(y) < 100:
            return None

        S = np.abs(librosa.stft(y))
        y_harm = librosa.effects.harmonic(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        features = {
            "path": blob_name,
            "filename": blob_name.split("/")[-1],
            "duration": float(len(y)) / float(sr),
        }

        stem = features["filename"].replace(".low.wav", "")
        features["track_num"] = int(stem) if stem.isdigit() else None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            m, s, mn, mx = _stats_1d(mfcc[i])
            features[f"mfcc_{i+1}_mean"] = m
            features[f"mfcc_{i+1}_std"] = s
            features[f"mfcc_{i+1}_min"] = mn
            features[f"mfcc_{i+1}_max"] = mx

        for name, arr in [
            ("centroid",  librosa.feature.spectral_centroid(S=S, sr=sr)[0]),
            ("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]),
            ("rolloff",   librosa.feature.spectral_rolloff(S=S, sr=sr)[0]),
            ("flatness",  librosa.feature.spectral_flatness(S=S)[0]),
        ]:
            m, s, mn, mx = _stats_1d(arr)
            features[f"{name}_mean"] = m
            features[f"{name}_std"] = s
            features[f"{name}_min"] = mn
            features[f"{name}_max"] = mx

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(7):
            m, s, mn, mx = _stats_1d(contrast[i])
            features[f"contrast_{i+1}_mean"] = m
            features[f"contrast_{i+1}_std"] = s
            features[f"contrast_{i+1}_min"] = mn
            features[f"contrast_{i+1}_max"] = mx

        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        features["tempo"] = float(np.atleast_1d(tempo)[0])

        m, s, mn, mx = _stats_1d(onset_env)
        features["onset_mean"] = m
        features["onset_std"] = s
        features["onset_min"] = mn
        features["onset_max"] = mx

        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        for i in range(12):
            m, s, mn, mx = _stats_1d(chroma[i])
            features[f"chroma_{i+1}_mean"] = m
            features[f"chroma_{i+1}_std"] = s
            features[f"chroma_{i+1}_min"] = mn
            features[f"chroma_{i+1}_max"] = mx

        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(6):
            m, s, mn, mx = _stats_1d(tonnetz[i])
            features[f"tonnetz_{i+1}_mean"] = m
            features[f"tonnetz_{i+1}_std"] = s
            features[f"tonnetz_{i+1}_min"] = mn
            features[f"tonnetz_{i+1}_max"] = mx

        for name, arr in [
            ("rms", librosa.feature.rms(S=S)[0]),
            ("zcr", librosa.feature.zero_crossing_rate(y)[0]),
        ]:
            m, s, mn, mx = _stats_1d(arr)
            features[f"{name}_mean"] = m
            features[f"{name}_std"] = s
            features[f"{name}_min"] = mn
            features[f"{name}_max"] = mx

        return features

    except Exception as e:
        tqdm.write(f"Error {blob_name}: {e}")
        return None


# -------------------------
# MAIN
# -------------------------

def main():
    # Parse args: python build_dataset.py 0 2  (split 0 of 2)
    #             python build_dataset.py 1 2  (split 1 of 2)
    split_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    split_total = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    out_blob = f"features/training_dataset/dataset_part{split_index}.parquet"

    print(f"Split {split_index + 1}/{split_total} — output: {out_blob}")

    meta_df = load_metadata()

    print("Listing WAV blobs...")
    all_blobs = sorted(
        b.name for b in client_global.list_blobs(BUCKET, prefix=WAV_PREFIX)
        if b.name.endswith(".wav")
    )

    # Each instance takes its slice
    blobs = all_blobs[split_index::split_total]
    print(f"Processing {len(blobs)} / {len(all_blobs)} files")

    all_records = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(extract_features, b): b for b in blobs}
        with tqdm(total=len(blobs), desc=f"Split {split_index}") as bar:
            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_records.append(result)
                bar.update(1)

    print(f"Extracted {len(all_records)} / {len(blobs)} successfully")

    features_df = pd.DataFrame(all_records)
    dataset = features_df.merge(meta_df, on="track_num", how="inner")

    print(f"Dataset shape: {dataset.shape}")

    print(f"Uploading to gs://{BUCKET}/{out_blob}...")
    buf = io.BytesIO()
    dataset.to_parquet(buf, index=False)
    buf.seek(0)
    client_global.bucket(BUCKET).blob(out_blob).upload_from_file(
        buf, content_type="application/octet-stream"
    )

    print(f"Done — {len(dataset)} rows uploaded to {out_blob}")


if __name__ == "__main__":
    main()