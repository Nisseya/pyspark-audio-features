import os
import warnings
import numpy as np
import librosa

import dotenv

dotenv.load_dotenv()

import sys

_hadoop_home = os.getenv("HADOOP_HOME")
if _hadoop_home:
    os.environ["PATH"] = _hadoop_home + r"\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from tqdm import tqdm

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, floor, row_number, sum as Fsum
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql.functions import udf

warnings.filterwarnings("ignore")

# spark = (
#     SparkSession.builder
#     .appName("AudioFeaturesCompactWithTQDM")
#     .master("local[*]")
#     .config("spark.driver.memory", "8g")
#     .config("spark.sql.shuffle.partitions", "16")
#     .getOrCreate()
# )
# spark.sparkContext.setLogLevel("ERROR")

# =========================
# Helpers : stats compactes
# =========================
from utils.features import _stats_1d, _stats_2d_rows

# =========================
# Schéma compact : scalaires
# =========================
fields = []

for i in range(20):
    fields += [
        StructField(f"mfcc_{i+1}_mean", DoubleType()),
        StructField(f"mfcc_{i+1}_std", DoubleType()),
        StructField(f"mfcc_{i+1}_min", DoubleType()),
        StructField(f"mfcc_{i+1}_max", DoubleType()),
    ]

for name in ["centroid", "bandwidth", "rolloff", "flatness"]:
    fields += [
        StructField(f"{name}_mean", DoubleType()),
        StructField(f"{name}_std", DoubleType()),
        StructField(f"{name}_min", DoubleType()),
        StructField(f"{name}_max", DoubleType()),
    ]

for i in range(7):
    fields += [
        StructField(f"contrast_{i+1}_mean", DoubleType()),
        StructField(f"contrast_{i+1}_std", DoubleType()),
        StructField(f"contrast_{i+1}_min", DoubleType()),
        StructField(f"contrast_{i+1}_max", DoubleType()),
    ]

fields.append(StructField("tempo", DoubleType()))

fields += [
    StructField("onset_mean", DoubleType()),
    StructField("onset_std", DoubleType()),
    StructField("onset_min", DoubleType()),
    StructField("onset_max", DoubleType()),
]

for i in range(12):
    fields += [
        StructField(f"chroma_{i+1}_mean", DoubleType()),
        StructField(f"chroma_{i+1}_std", DoubleType()),
        StructField(f"chroma_{i+1}_min", DoubleType()),
        StructField(f"chroma_{i+1}_max", DoubleType()),
    ]

for i in range(6):
    fields += [
        StructField(f"tonnetz_{i+1}_mean", DoubleType()),
        StructField(f"tonnetz_{i+1}_std", DoubleType()),
        StructField(f"tonnetz_{i+1}_min", DoubleType()),
        StructField(f"tonnetz_{i+1}_max", DoubleType()),
    ]

for name in ["rms", "zcr"]:
    fields += [
        StructField(f"{name}_mean", DoubleType()),
        StructField(f"{name}_std", DoubleType()),
        StructField(f"{name}_min", DoubleType()),
        StructField(f"{name}_max", DoubleType()),
    ]

fields.append(StructField("ok", IntegerType()))
schema = StructType(fields)

# =========================
# UDF : compact features
# =========================
def extract_features_compact(path: str):
    clean_path = path.replace("file:", "") if path.startswith("file:") else path
    try:
        y, sr = librosa.load(clean_path, sr=None, mono=True)
        if y is None or len(y) < 100:
            return tuple([0.0] * (len(fields) - 1) + [0])

        S = np.abs(librosa.stft(y))
        y_harm = librosa.effects.harmonic(y)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)

        out = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        out += _stats_2d_rows(mfcc, 20)

        centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        for v in [centroid, bandwidth, rolloff, flatness]:
            out += list(_stats_1d(v))

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        out += _stats_2d_rows(contrast, 7)

        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        out.append(float(np.atleast_1d(tempo)[0]))

        out += list(_stats_1d(onset_env))

        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        out += _stats_2d_rows(chroma, 12)

        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        out += _stats_2d_rows(tonnetz, 6)

        rms = librosa.feature.rms(S=S)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        out += list(_stats_1d(rms))
        out += list(_stats_1d(zcr))

        out.append(1)
        return tuple(out)

    except Exception:
        return tuple([0.0] * (len(fields) - 1) + [0])

extract_udf = udf(extract_features_compact, schema)

# =========================
# Data : paths wav
# =========================

if __name__ == "__main__":
#  Added Session here
    spark = (
        SparkSession.builder
        .appName("AudioFeaturesCompactWithTQDM")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    audio_root = "data/audio/wav"
    out_path = "data/features/parquet_compact"

    paths_df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.wav")
        .option("recursiveFileLookup", "true")
        .load(audio_root)
        .select("path")
        .dropDuplicates(["path"])
    )
    # =========================
    # Batching stable (row_number)
    # =========================
    BATCH_SIZE = 200  # ajuste: 100-500 selon RAM/CPU

    w = Window.orderBy("path")
    paths_df = paths_df.withColumn("rn", row_number().over(w) - 1)
    paths_df = paths_df.withColumn("batch_id", floor(col("rn") / lit(BATCH_SIZE)).cast("long")).drop("rn")

    batch_ids = [r["batch_id"] for r in paths_df.select("batch_id").distinct().orderBy("batch_id").collect()]
    print("BATCHES:", len(batch_ids), "| BATCH_SIZE:", BATCH_SIZE)
    print("OUT:", os.path.abspath(out_path))

    spark._jsc.hadoopConfiguration().set("parquet.enable.summary-metadata", "false")

    ok_total = 0
    ko_total = 0

    for bid in tqdm(batch_ids, desc="Batches", unit="batch"):
        batch = paths_df.filter(col("batch_id") == lit(bid)).repartition(8)
        if batch.rdd.isEmpty():
            continue

        feat_df = (
            batch
            .withColumn("f", extract_udf(col("path")))
            .select("path", "batch_id", "f.*")
        )

        total = paths_df.count()
        print("TOTAL WAV:", total)

        # =========================
        # Batching stable (row_number)
        # =========================
        BATCH_SIZE = 200  # ajuste: 100-500 selon RAM/CPU

        w = Window.orderBy("path")
        paths_df = paths_df.withColumn("rn", row_number().over(w) - 1)
        paths_df = paths_df.withColumn("batch_id", floor(col("rn") / lit(BATCH_SIZE)).cast("long")).drop("rn")

        batch_ids = [r["batch_id"] for r in paths_df.select("batch_id").distinct().orderBy("batch_id").collect()]
        print("BATCHES:", len(batch_ids), "| BATCH_SIZE:", BATCH_SIZE)
        print("OUT:", os.path.abspath(out_path))

    spark._jsc.hadoopConfiguration().set("parquet.enable.summary-metadata", "false")

    ok_total = 0
    ko_total = 0

    for bid in tqdm(batch_ids, desc="Batches", unit="batch"):
        batch = paths_df.filter(col("batch_id") == lit(bid)).repartition(8)
        if batch.rdd.isEmpty():
            continue

        feat_df = (
            batch
            .withColumn("f", extract_udf(col("path")))
            .select("path", "batch_id", "f.*")
        )

        c = feat_df.select(
            Fsum(col("ok")).alias("ok"),
            Fsum((1 - col("ok"))).alias("ko")
        ).collect()[0]

        ok_b = int(c["ok"])
        ko_b = int(c["ko"])
        ok_total += ok_b
        ko_total += ko_b

        (
            feat_df
            .where(col("ok") == 1)
            .drop("ok")
            .write
            .mode("append")
            .partitionBy("batch_id")
            .parquet(out_path)
        )

        tqdm.write(f"batch={bid} ok={ok_b} ko={ko_b} | cumul ok={ok_total} ko={ko_total}")

    print("DONE | OK:", ok_total, "KO:", ko_total)
    spark.stop()
