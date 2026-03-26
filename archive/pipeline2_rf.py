
import os, sys
os.environ["PYSPARK_PYTHON"] = sys.executable

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, array_contains
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(os.path.join(PROJECT_ROOT, "data/features/training_dataset"))

df = (
    df
    .filter(col("genres").isNotNull())
    .filter(size(col("genres")) > 0)
)

excluded = {"path", "TRACK_ID", "meta_path", "genres"}
feature_cols = [
    name for name, dtype in df.dtypes
    if dtype in ("double", "int", "bigint", "float") and name not in excluded
]

genre_counts = (
    df
    .selectExpr("explode(genres) as genre")
    .groupBy("genre")
    .count()
    .filter(col("count") >= 20)
    .orderBy(col("count").desc())
)

genres = [row["genre"] for row in genre_counts.collect()]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

df = assembler.transform(df).select("track_num", "genres", "features")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

models = {}
scores = []

for genre in genres:
    train_bin = train_df.withColumn("label", array_contains(col("genres"), genre).cast("double"))
    test_bin = test_df.withColumn("label", array_contains(col("genres"), genre).cast("double"))

    pos = train_bin.filter(col("label") == 1.0).count()
    neg = train_bin.filter(col("label") == 0.0).count()

    if pos == 0 or neg == 0:
        continue

    model = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=100,
        maxDepth=10,
        seed=42
    ).fit(train_bin)

    pred = model.transform(test_bin)

    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="probability",
        metricName="areaUnderROC"
    ).evaluate(pred)

    models[genre] = model
    scores.append((genre, pos, neg, auc))

for genre, pos, neg, auc in scores:
    print(f"{genre:20s} pos={pos:5d} neg={neg:5d} auc={auc:.4f}")
