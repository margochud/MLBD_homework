from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import DecisionTreeClassifier, LogisticRegression, NaiveBayes, LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, Imputer, StringIndexer, MinMaxScaler, Normalizer
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import os

DIR = "/home/margarita/HSE/2course/MLBD/task2/data"
def set_up_env():
    os.environ["SPARK_HOME"] = "/home/margarita/spark"
    os.environ["PYSPARK_PYTHON"] = "/home/margarita/anaconda3/envs/pythonProject/bin/python"

def getSparkSession():
    set_up_env()

    spark = SparkSession \
        .builder \
        .master("local[2]") \
        .appName("Titanic") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

class Printer(Transformer):
    def __init__(self, message: str):
        super(Printer, self).__init__()
        self.message = message

    def _transform(self, dataset: DataFrame) -> DataFrame:
        print(self.message)
        dataset.show(truncate=False)
        dataset.printSchema()
        return dataset

def evaluation(pipeline, training, test):
    model = pipeline.fit(training)
    rawPredictions = model.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol="classIndexed", predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(rawPredictions)
    return accuracy

def print_acc(item):
    accuracy, name = item
    print(f"Test Error for {name}= {(1.0 - accuracy)}")


if __name__ == "__main__":
    set_up_env()
    spark = getSparkSession()

    mushrooms = spark.read \
        .option("delimiter", ",") \
        .option("inferSchema", "true") \
        .option("header", "true") \
        .csv("./mushrooms.csv")
    mushrooms.printSchema()
    columns = mushrooms.columns
    #все колонки в датасете категориальные (включая лейблы), преобразуем их
    for col in columns:
        NewIndexer = StringIndexer(inputCol=col,
                                   outputCol=col+"Indexed",
                                   handleInvalid="keep")
        mushrooms = NewIndexer.fit(mushrooms).transform(mushrooms)
    mushrooms = mushrooms.drop(*columns)
    #print(mushrooms.select('classIndexed').distinct().collect())
    mushrooms.show()

    #разбиваем получившиеся данные на train и test
    training, test = mushrooms.randomSplit([0.7, 0.3], seed=12345)
    training.cache()
    test.cache()

    #дальнейшие шаги запишем в Pipeline

    inp_in = mushrooms.columns #все колонки кроме class
    inp_out = [name+"imputed" for name in mushrooms.columns]
    imputer = Imputer(strategy="mode",
                      inputCols=inp_in,
                      outputCols=inp_out)

    assembler = VectorAssembler(
        inputCols=inp_out[1:],
        outputCol="features")

    #проверим модели, предложенные в домашнем задании
    trainer_log = LogisticRegression(labelCol="classIndexedimputed", featuresCol="features")
    trainer_NB = NaiveBayes(labelCol="classIndexedimputed", featuresCol="features")
    trainer_SVC = LinearSVC(labelCol="classIndexedimputed", featuresCol="features")
    trainer_RF = RandomForestClassifier(labelCol="classIndexedimputed", featuresCol="features")

    pipeline_log = Pipeline(stages=[imputer,
                                assembler,
                               trainer_log])

    pipeline_NB = Pipeline(stages=[imputer,
                                assembler,
                               trainer_NB])
    pipeline_SVC = Pipeline(stages=[imputer,
                                    assembler,
                                    trainer_SVC])
    pipeline_RF = Pipeline(stages=[imputer,
                                    assembler,
                                    #Printer("After Final Assembling"),
                                    trainer_RF])

    history = []
    history.append([evaluation(pipeline_log, training, test), "Logistic Regression"])
    history.append([evaluation(pipeline_NB, training, test), "Naive Bayes"])
    history.append([evaluation(pipeline_RF, training, test), "Random Forest"])
    history.append([evaluation(pipeline_SVC, training, test), "SVC"])

    for item in history:
        print_acc(item)
    #все работает неплохо, но Random Forest лучше всех
    #сделаем кросс-валидацию для лучшей модели
    #будем менять количество деревьев и их глубину

    paramGrid = ParamGridBuilder() \
        .addGrid(trainer_RF.numTrees, [5, 10, 20, 30]) \
        .addGrid(trainer_RF.maxDepth, [3, 5, 7]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol="classIndexed", predictionCol="prediction",
                                                  metricName="accuracy")

    cv = CrossValidator(estimator=pipeline_RF,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3)

    cvModel = cv.fit(training)
    print()
    print("Best model is", cvModel.bestModel)
    print(f"Average metrics: {', '.join([str(i) for i in cvModel.avgMetrics])}")

    rawPredictions = cvModel.transform(test)
    print("Best num of trees: " + str(cvModel.bestModel.stages[2].getNumTrees))
    print("Best depth: " + str(cvModel.bestModel.stages[2].getMaxDepth()))

    acc = evaluator.evaluate(rawPredictions)
    print(f"Test accuracy: {acc}")

    spark.stop()
