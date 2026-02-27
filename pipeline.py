from pyspark.sql import SparkSession

from pyspark.sql import SparkSession

spark = (
    SparkSession
        .builder
        .appName("MonApplication")
        .master("local[4]")
        .config("spark.driver.memory", "15g")
        .config("spark.executor.memory","15g")
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", "200")
        .getOrCreate()
)


sc = spark.sparkContext
sc.setLogLevel("WARN")


df = spark.read.format("binaryFile") \
          .option("pathGlobFilter", "*.wav") \
          .option("recursiveFileLookup", "true") \
          .load("./data/audio/wav/")  

print("Schema:")
df.printSchema()

print("Count of files:")
print(df.count())