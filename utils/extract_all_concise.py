import os
import warnings
import numpy as np
import librosa
import dotenv
import sys

dotenv.load_dotenv()

_hadoop_home = os.getenv("HADOOP_HOME")
if _hadoop_home:
    os.environ["PATH"] = _hadoop_home + r"\bin;" + os.environ.get("PATH", "")
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, floor, row_number
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.sql.functions import udf

warnings.filterwarnings("ignore")

from utils.features import _stats_1d, _stats_2d_rows

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


def extract_features_compact(path: str):
    import librosa
    import numpy as np
    from utils.features import _stats_1d, _stats_2d_rows

    clean_path = path.replace("file:///", "").replace("file://", "").replace("file:/", "").replace("\\", "/")
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

        for arr in [
            librosa.feature.spectral_centroid(S=S, sr=sr)[0],
            librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
            librosa.feature.spectral_rolloff(S=S, sr=sr)[0],
            librosa.feature.spectral_flatness(S=S)[0],
        ]:
            out += list(_stats_1d(arr))

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        out += _stats_2d_rows(contrast, 7)

        tempo = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)[0]
        out.append(float(np.atleast_1d(tempo)[0]))

        out += list(_stats_1d(onset_env))

        chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr)
        out += _stats_2d_rows(chroma, 12)

        tonnetz = librosa.feature.tonnetz(y=y_harm, sr=sr)
        out += _stats_2d_rows(tonnetz, 6)

        out += list(_stats_1d(librosa.feature.rms(S=S)[0]))
        out += list(_stats_1d(librosa.feature.zero_crossing_rate(y)[0]))

        out.append(1)
        return tuple(out)

    except Exception:
        return tuple([0.0] * (len(fields) - 1) + [0])


extract_udf = udf(extract_features_compact, schema)

if __name__ == "__main__":
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
    BATCH_SIZE = 200

    spark._jsc.hadoopConfiguration().set("parquet.enable.summary-metadata", "false")

    w = Window.orderBy("path")

    (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.wav")
        .option("recursiveFileLookup", "true")
        .load(audio_root)
        .select("path")
        .dropDuplicates(["path"])
        .withColumn("batch_id", floor((row_number().over(w) - 1) / lit(BATCH_SIZE)).cast("long"))
        .repartition(8, "batch_id")
        .withColumn("f", extract_udf(col("path")))
        .select("path", "batch_id", "f.*")
        .where(col("ok") == 1)
        .drop("ok")
        .write
        .mode("append")
        .partitionBy("batch_id")
        .parquet(out_path)
    )

    spark.stop()