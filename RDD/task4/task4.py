from pyspark.sql import SparkSession
import os
DIR = "/home/margarita/HSE/2course/MLBD/task2/data"
def set_up_env():
    os.environ["SPARK_HOME"] = "/home/margarita/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/margarita/anaconda3/envs/pythonProject/bin/python"

def mean_salary(salaries):
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("task_2") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext
    #превращаем данные в rdd
    tup = [tuple([item.split()[0], (int(item.split()[1]))]) for item in salaries]
    rdd = sc.parallelize(tup)
    #считаем среднее
    print(rdd.mapValues(lambda v: (v, 1)).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1])) \
          .mapValues(lambda v: v[0]/v[1]).collectAsMap())
    spark.stop()

if __name__ == "__main__":

    salaries = [
        "John 1900 January",
        "Mary 2000 January",
        "John 1800 February",
        "John 1000 March",
        "Mary 1500 February",
        "Mary 2900 March",
        "Mary 1600 April",
        "John 2800 April"
    ]
    mean_salary(salaries)