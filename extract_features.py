import os
print(os.getcwd())

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, DoubleType
import librosa
import numpy as np

spark = SparkSession.builder \
    .appName("AudioFeatureExtraction") \
    .master("local[*]") \
    .getOrCreate()

data = [("../809.low.mp3",)]
df = spark.createDataFrame(data, ["path"])


# =========================
# Schema dynamique
# =========================

feature_fields = []

# MFCC 20 mean + 20 std
for i in range(20):
    feature_fields.append(StructField(f"mfcc_{i+1}_mean", DoubleType()))
for i in range(20):
    feature_fields.append(StructField(f"mfcc_{i+1}_std", DoubleType()))

# Spectral features
feature_fields += [
    StructField("centroid_mean", DoubleType()),
    StructField("centroid_std", DoubleType()),
    StructField("bandwidth_mean", DoubleType()),
    StructField("bandwidth_std", DoubleType()),
    StructField("rolloff_mean", DoubleType()),
    StructField("rolloff_std", DoubleType()),
    StructField("flatness_mean", DoubleType()),
    StructField("flatness_std", DoubleType()),
]

# Spectral contrast (7 bandes par défaut)
for i in range(7):
    feature_fields.append(StructField(f"contrast_{i+1}_mean", DoubleType()))
for i in range(7):
    feature_fields.append(StructField(f"contrast_{i+1}_std", DoubleType()))

# Tempo + Onset
feature_fields += [
    StructField("tempo", DoubleType()),
    StructField("onset_mean", DoubleType()),
    StructField("onset_std", DoubleType()),
]

# Chroma (12)
for i in range(12):
    feature_fields.append(StructField(f"chroma_{i+1}_mean", DoubleType()))
for i in range(12):
    feature_fields.append(StructField(f"chroma_{i+1}_std", DoubleType()))

# Tonnetz (6)
for i in range(6):
    feature_fields.append(StructField(f"tonnetz_{i+1}_mean", DoubleType()))
for i in range(6):
    feature_fields.append(StructField(f"tonnetz_{i+1}_std", DoubleType()))

# RMS + ZCR
feature_fields += [
    StructField("rms_mean", DoubleType()),
    StructField("rms_std", DoubleType()),
    StructField("zcr_mean", DoubleType()),
    StructField("zcr_std", DoubleType()),
]

schema = StructType(feature_fields)


# =========================
# Feature extraction
# =========================

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=None)
        values = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        values.extend(np.mean(mfcc, axis=1))
        values.extend(np.std(mfcc, axis=1))

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)

        values += [
            np.mean(centroid), np.std(centroid),
            np.mean(bandwidth), np.std(bandwidth),
            np.mean(rolloff), np.std(rolloff),
            np.mean(flatness), np.std(flatness),
        ]

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        values.extend(np.mean(contrast, axis=1))
        values.extend(np.std(contrast, axis=1))

        # Tempo & onset
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        values += [
            float(tempo),
            np.mean(onset_env),
            np.std(onset_env),
        ]

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        values.extend(np.mean(chroma, axis=1))
        values.extend(np.std(chroma, axis=1))

        # Tonnetz
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        values.extend(np.mean(tonnetz, axis=1))
        values.extend(np.std(tonnetz, axis=1))

        # RMS + ZCR
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)

        values += [
            np.mean(rms), np.std(rms),
            np.mean(zcr), np.std(zcr),
        ]

        return tuple(float(v) for v in values)

    except Exception as e:
        print("Erreur:", e)
        return tuple([0.0] * len(feature_fields))


extract_udf = udf(extract_features, schema)

df_features = df.select("path", extract_udf("path").alias("features")).select("path", "features.*")

df_features.show(truncate=False)