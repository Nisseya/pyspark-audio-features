from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, array_contains
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/features/training_dataset")

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
    outputCol="features_raw",
    handleInvalid="skip"
)

df = assembler.transform(df).select("track_num", "genres", "features_raw")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

scaler = StandardScaler(
    inputCol="features_raw",
    outputCol="features",
    withMean=True,
    withStd=True
)

scaler_model = scaler.fit(train_df)
train_df = scaler_model.transform(train_df).select("track_num", "genres", "features")
test_df = scaler_model.transform(test_df).select("track_num", "genres", "features")

models = {}
scores = []

for genre in genres:
    train_bin = train_df.withColumn("label", array_contains(col("genres"), genre).cast("double"))
    test_bin = test_df.withColumn("label", array_contains(col("genres"), genre).cast("double"))

    pos = train_bin.filter(col("label") == 1.0).count()
    neg = train_bin.filter(col("label") == 0.0).count()

    if pos == 0 or neg == 0:
        continue

    model = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1
    ).fit(train_bin)

    pred = model.transform(test_bin)

    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(pred)

    models[genre] = model
    scores.append((genre, pos, neg, auc))

for genre, pos, neg, auc in scores:
    print(f"{genre:20s} pos={pos:5d} neg={neg:5d} auc={auc:.4f}")