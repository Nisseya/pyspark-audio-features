from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, element_at
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/features/training_dataset")

df = (
    df
    .filter(col("genres").isNotNull())
    .filter(size(col("genres")) > 0)
    .withColumn("label_str", element_at(col("genres"), 1))
)

excluded = {"path", "TRACK_ID", "meta_path", "genres", "label_str"}
feature_cols = [
    name for name, dtype in df.dtypes
    if dtype in ("double", "int", "bigint", "float") and name not in excluded
]

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

pipeline = Pipeline(stages=[
    StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="skip"),
    VectorAssembler(inputCols=feature_cols, outputCol="features_raw", handleInvalid="skip"),
    StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True),
    LogisticRegression(
        featuresCol="features",
        labelCol="label",
        maxIter=100,
        regParam=0.1,
        family="multinomial"
    )
])

model = pipeline.fit(train_df)
pred = model.transform(test_df)

accuracy = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
).evaluate(pred)

f1 = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
).evaluate(pred)

print("Accuracy:", accuracy)
print("F1:", f1)

pred.select("label_str", "prediction", "probability").show(20, truncate=False)