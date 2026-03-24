import os
import warnings
import numpy as np
import librosa
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, ArrayType, DoubleType

# Suppression des logs inutiles
warnings.filterwarnings("ignore")

# Configuration Spark pour le parallélisme local
spark = (
    SparkSession.builder
    .appName("TurboAudioFeatures")
    .master("local[*]")  # Utilise tous les cœurs
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.driver.memory", "8g") # Augmenté pour éviter les crashs
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# --- Schéma (identique au tien) ---
schema = StructType([
    *[StructField(f"mfcc_{i+1}", ArrayType(DoubleType())) for i in range(20)],
    StructField("spectral_centroid", ArrayType(DoubleType())),
    StructField("spectral_bandwidth", ArrayType(DoubleType())),
    StructField("spectral_rolloff", ArrayType(DoubleType())),
    StructField("spectral_flatness", ArrayType(DoubleType())),
    *[StructField(f"spectral_contrast_{i+1}", ArrayType(DoubleType())) for i in range(7)],
    StructField("tempo", DoubleType()),
    StructField("onset_strength", ArrayType(DoubleType())),
    *[StructField(f"chroma_{i+1}", ArrayType(DoubleType())) for i in range(12)],
    *[StructField(f"tonnetz_{i+1}", ArrayType(DoubleType())) for i in range(6)],
    StructField("rms", ArrayType(DoubleType())),
    StructField("zcr", ArrayType(DoubleType())),
])

def extract_features_v2(path: str):
    """Fonction optimisée : calcule la STFT une seule fois pour tout le monde"""
    clean_path = path.replace("file:", "") if path.startswith("file:") else path
    
    try:
        # Chargement rapide (mono, sr native)
        y, sr = librosa.load(clean_path, sr=None)
        if len(y) < 100: return None

        # --- OPTIMISATION CPU : Calculer les bases une seule fois ---
        S = np.abs(librosa.stft(y))
        y_harm = librosa.effects.harmonic(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        vals = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20): vals.append(mfcc[i, :].tolist())

        # Spectral (utilise S pré-calculé)
        vals.append(librosa.feature.spectral_centroid(S=S, sr=sr)[0].tolist())
        vals.append(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0].tolist())
        vals.append(librosa.feature.spectral_rolloff(S=S, sr=sr)[0].tolist())
        vals.append(librosa.feature.spectral_flatness(S=S)[0].tolist())

        # Contrast
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(7): vals.append(contrast[i, :].tolist())

        # Tempo & Onset
        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        vals.append(float(np.atleast_1d(tempo)[0]))
        vals.append(onset_env.astype(float).tolist())

        # Chroma & Tonnetz (utilise y_harm)
        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        for i in range(12): vals.append(chroma[i, :].tolist())

        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        for i in range(6): vals.append(tonnetz[i, :].tolist())

        # RMS & ZCR
        vals.append(librosa.feature.rms(S=S)[0].tolist())
        vals.append(librosa.feature.zero_crossing_rate(y)[0].tolist())

        return tuple(vals)
    except:
        return None

extract_udf = udf(extract_features_v2, schema)

# --- Exécution ---
audio_root = "data/audio/wav"
out_path = "data/features/parquet"

print(">>> Analyse des fichiers...")
paths_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.wav")
    .load(audio_root)
    .select("path")
    # REPARTITION : C'est ici qu'on débloque le CPU. 
    # On crée 2 à 4 partitions par cœur physique disponible.
    .repartition(32) 
)

print(">>> Extraction en cours (Sature le CPU)...")
(
    paths_df
    .withColumn("f", extract_udf(col("path")))
    .where(col("f").isNotNull())
    .select("path", "f.*")
    .write
    .mode("overwrite")
    .parquet(out_path)
)

print(">>> Terminé !")
spark.stop()