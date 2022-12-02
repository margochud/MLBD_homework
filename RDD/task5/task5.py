from pyspark.sql import SparkSession
from pyspark.mllib.stat import Statistics
from math import sqrt
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

    data = [-1, 1, 3, 2, 2, 150, 1, 2, 3, 2, 2, 1, 1, 1, -100, 2, 2, 3, 4, 1, 1, 3, 4]
    data = [float(i) for i in data]
    rdd = sc.parallelize(data)
    #считаем длину выборки, сумму элементов и их квадратов
    params = (rdd
             .map(lambda x: (1, x, x * x))
             .reduce(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2])))
    #находим среднее и среднеквадратичное отклонение
    sigma = sqrt(params[2]/params[0] - (params[1]/params[0])**2)
    mean = params[1]/params[0]
    print(f"mean = {mean}, sigma = {sigma}")
    # убираем ненужные данные
    data_tr = rdd.filter(lambda x: abs(x-mean) < 3*sigma).collect()
    print(f"Data for Vasya: {data_tr}")
    #проверяем выборку на нормальность по критерию Колмогорова-Смирнова
    print(Statistics.kolmogorovSmirnovTest(rdd, "norm"))

    spark.stop()