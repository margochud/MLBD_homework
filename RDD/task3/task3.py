from pyspark.sql import SparkSession
import os
DIR = "/home/margarita/HSE/2course/MLBD/task2/data"
def set_up_env():
    os.environ["SPARK_HOME"] = "/home/margarita/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/margarita/anaconda3/envs/pythonProject/bin/python"

if __name__ == "__main__":
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("task_2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext

    jvmLanguages = sc.parallelize(["Scala", "Java", "Groovy", "Kotlin", "Ceylon"])
    functionalLanguages = sc.parallelize(["Scala", "Kotlin", "JavaScript", "Haskell", "Python"])
    webLanguages = sc.parallelize(["PHP", "Ruby", "Perl", "JavaScript", "Python"])
    mlLanguages = sc.parallelize(["JavaScript", "Python", "Scala"])

    print(f"ML and JVM : {', '.join(jvmLanguages.intersection(mlLanguages).collect())}")
    print(f"Web, but not functional : {', '.join(webLanguages.subtract(functionalLanguages).collect())}")
    print(f"JVM and functional : {', '.join(jvmLanguages.union(mlLanguages).collect())}")

    spark.stop()