from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import os
import math
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

    #создаем RDD
    ds = range(1, 100000)
    rdd = sc.parallelize(ds, 10)
    #сохраняем в текстовые файлы
    if os.path.exists(DIR):
        for name in os.listdir(DIR):
            os.remove(os.path.join(DIR, name))
        os.rmdir(DIR)
    rdd.saveAsTextFile(DIR)
    #считываем обратно
    rdd2 = sc.textFile(DIR)
    #создаем пары и делим на группы
    groups = rdd2.map(lambda x: (float(x) % 100, math.log(float(x))))
    res = groups.filter(lambda x: (int(10*x[1]) % 10) % 2 == 0).groupByKey().mapValues(len).collect()
    #выводим количество элементов для каждого ключа
    for key, l in res:
        print(f"{key}: {l}")
    spark.stop()