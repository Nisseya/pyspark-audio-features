
import os, sys
import dotenv

dotenv.load_dotenv()

if os.getenv("CESTQUIQUIADESPROBLEMESAVECSPARK") == "Leo":
    os.environ["PYSPARK_PYTHON"] = r"C:\spark-env\Scripts\python.exe"
    os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\spark-env\Scripts\python.exe"
    os.environ["HADOOP_HOME"] = r"C:\hadoop"
    os.environ["PATH"] = r"C:\hadoop\bin;" + os.environ["PATH"]

# _hadoop_home = os.getenv("HADOOP_HOME")
# if _hadoop_home:
#     os.environ["PATH"] = _hadoop_home + r"\bin;" + os.environ.get("PATH", "")
# os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
# os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, element_at
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = (
    SparkSession.builder
    .appName("RandomForestPipeline")
    .config("spark.driver.memory", "10g")
    .config("spark.driver.memoryOverhead", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.memory.fraction", "0.8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(os.path.join(PROJECT_ROOT, "data/features/training_dataset"))

df = (
    df
    .filter(col("genres").isNotNull())
    .filter(size(col("genres")) > 0)
    .withColumn("label_str", element_at(col("genres"), 1))
)

excluded = {"path", "TRACK_ID", "meta_path", "genres", "label_str", "track_num", "batch_id"}
feature_cols = [
    name for name, dtype in df.dtypes
    if dtype in ("double", "int", "bigint", "float") and name not in excluded
]

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

pipeline = Pipeline(stages=[
    StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="skip"),
    VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip"),
    RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        numTrees=75,
        maxDepth=10,
        maxBins=32,
        minInstancesPerNode=5,
        seed=42
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

model_path = os.path.join(PROJECT_ROOT, "data/model/rf_pipeline")
model.write().overwrite().save(model_path)
print("Modèle sauvegardé :", model_path)