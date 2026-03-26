from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, floor, rand
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

BUCKET = "spark-audio-bucket"
features_path = f"gs://{BUCKET}/features/parquet_compact"
tsv_path = f"gs://{BUCKET}/assets/raw_30s.tsv"
out_path = f"gs://{BUCKET}/features/training_dataset"

features_df = spark.read.parquet(features_path)

features_df = features_df.withColumn(
    "track_num",
    regexp_extract(col("path"), r"([^/]+)\.low\.wav$", 1).cast("int")
)

meta_df = (
    spark.read
    .option("header", True)
    .option("sep", "\t")
    .csv(tsv_path)
)

meta_df = (
    meta_df
    .withColumn(
        "track_num",
        regexp_extract(col("TRACK_ID"), r"track_0*([0-9]+)$", 1).cast("int")
    )
    .withColumn(
        "genres",
        F.regexp_extract_all(col("TAGS"), F.lit(r"genre---([^\t]+)"), F.lit(1))
    )
    .select(
        col("TRACK_ID"),
        col("PATH").alias("meta_path"),
        col("track_num"),
        col("genres")
    )
)

train_df = (
    features_df
    .join(meta_df, on="track_num", how="inner")
    .withColumn("user_id", floor(rand() * 6).cast("int"))
)

train_df.select("track_num", "path", "filename", "duration", "user_id", "TRACK_ID", "genres").show(5, truncate=False)

train_df.write.mode("overwrite").parquet(out_path)