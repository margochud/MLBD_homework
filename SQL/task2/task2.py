from __future__ import print_function
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
import os

def set_up_env():
    os.environ["SPARK_HOME"] = "/home/margarita/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/margarita/anaconda3/envs/pythonProject/bin/python"

# Define UDF and use in DF API

def isWorldWarTwoYearFunction(year):
    return True if (1939 <= year <= 1945) else False


def isCanadaNeighbourFunction(stateName):
    canadaBorder = ["AK", "MI", "ME", "MN", "MT", "NY", "WA", "ND", "OH", "VT", "NH", "ID", "PA"]
    return True if (stateName in canadaBorder) else False


if __name__ == "__main__":
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("DataFrame Intro") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    stateNames = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .csv("StateNames.csv")

    stateNames.cache()

    stateNames.show()

    from pyspark.sql.types import BooleanType

    isWorldWarTwoYear = udf(lambda year: isWorldWarTwoYearFunction(year), BooleanType())
    isCanadaNeighbour = udf(lambda state: isCanadaNeighbourFunction(state), BooleanType())
    #сначала добавляем столбцы с нужными характеристиками
    boolData = stateNames.select("Name", "Gender", isWorldWarTwoYear(stateNames["Year"]),
                                 isCanadaNeighbour(stateNames["State"])).withColumnRenamed('<lambda>(Year)', 'isWW')\
        .withColumnRenamed('<lambda>(State)', 'isNeigh')
    boolData.show()
    boolData.createOrReplaceTempView("boolData")
    #оставляем имена, которые не повторяются и соответствуют условиями
    boarderNames = spark.sql("SELECT DISTINCT Name FROM boolData WHERE (Gender = 'F')"
              " AND (isWW = TRUE) AND (isNeigh = TRUE) ORDER BY Name")
    boarderNames.show()
    boarderNames.write.parquet("boarderNames")
    #для себя, посмотреть на данные
    boarderNames.write.csv("boarderNames.csv")
    spark.stop()
