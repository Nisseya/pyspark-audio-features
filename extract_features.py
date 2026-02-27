import os
print(os.getcwd())

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType
import librosa
import numpy as np

spark = SparkSession.builder \
    .appName("AudioFeatureExtraction") \
    .master("local[*]") \
    .getOrCreate()

data = [("../809.low.mp3",)]
df = spark.createDataFrame(data, ["path"])


# =========================
# Schema : chaque feature = tableau
# =========================

schema = StructType([

    # MFCC (20 coefficients)
    *[StructField(f"mfcc_{i+1}", ArrayType(DoubleType())) for i in range(20)],

    # Spectral features
    StructField("spectral_centroid", ArrayType(DoubleType())),
    StructField("spectral_bandwidth", ArrayType(DoubleType())),
    StructField("spectral_rolloff", ArrayType(DoubleType())),
    StructField("spectral_flatness", ArrayType(DoubleType())),

    # Spectral contrast (7 bandes)
    *[StructField(f"spectral_contrast_{i+1}", ArrayType(DoubleType())) for i in range(7)],

    # Rythme
    StructField("tempo", DoubleType()),  # tempo reste scalaire
    StructField("onset_strength", ArrayType(DoubleType())),

    # Harmonie
    *[StructField(f"chroma_{i+1}", ArrayType(DoubleType())) for i in range(12)],

    *[StructField(f"tonnetz_{i+1}", ArrayType(DoubleType())) for i in range(6)],

    # Dynamique
    StructField("rms", ArrayType(DoubleType())),
    StructField("zcr", ArrayType(DoubleType())),
])


# =========================
# Extraction
# =========================

def extract_features(path):
    try:
        y, sr = librosa.load(path, sr=None)

        values = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            values.append(mfcc[i, :].tolist())

        # Spectral
        values.append(librosa.feature.spectral_centroid(y=y, sr=sr)[0].tolist())
        values.append(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].tolist())
        values.append(librosa.feature.spectral_rolloff(y=y, sr=sr)[0].tolist())
        values.append(librosa.feature.spectral_flatness(y=y)[0].tolist())

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(7):
            values.append(contrast[i, :].tolist())

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        values.append(float(tempo))

        # Onset strength
        onset = librosa.onset.onset_strength(y=y, sr=sr)
        values.append(onset.tolist())

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        for i in range(12):
            values.append(chroma[i, :].tolist())

        # Tonnetz
        y_harm = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(6):
            values.append(tonnetz[i, :].tolist())

        # RMS
        values.append(librosa.feature.rms(y=y)[0].tolist())

        # ZCR
        values.append(librosa.feature.zero_crossing_rate(y)[0].tolist())

        return tuple(values)

    except Exception as e:
        print("Erreur:", e)
        return tuple([[] for _ in range(len(schema))])


extract_udf = udf(extract_features, schema)

df_features = df.select(
    "path",
    extract_udf("path").alias("features")
).select("path", "features.*")

df_features.show(truncate=False)
