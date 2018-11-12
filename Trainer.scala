package com.sparkProject

import org.apache.spark.SparkConf

import scala.collection.JavaConverters._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.ml.feature.{HashingTF, Tokenizer,  RegexTokenizer, StopWordsRemover, CountVectorizerModel,
  CountVectorizer, IDF, StringIndexer, OneHotEncoder, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.sql.functions._


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark: SparkSession = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3 Stephane Reynal
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("hello world ! from Trainer")

    // 1- Chargez le dataFrame obtenu à la fin du TP 2

    val df = spark.read.parquet("/home/stephane/Telecom/P1/INF729 - Hadoop/Spark/TP3/prepared_trainingset")

    // On modifie les appellations des pays en données numériques, dont la valeur est basée sur la frequence d'occurence
   val strIndexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed2")

    // Même chose pour les monnaies (currency)
    val strIndexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed2")

    // OneHotEncoder ajoute une colonne par catégorie de valeur (encodage binaire : 0 ou 1)
    val countryEncoder = new OneHotEncoder()
      .setInputCol("country_indexed2")
      .setOutputCol("country_indexed")

    // Même chose pour currency
    val currencyEncoder = new OneHotEncoder()
      .setInputCol("currency_indexed2")
      .setOutputCol("currency_indexed")

    val pipeline1 = new Pipeline()
      .setStages(Array(strIndexerCountry, strIndexerCurrency,
        countryEncoder,currencyEncoder))

    val model = pipeline1.fit(df)
    val dfprocessed = model.transform(df)

    //Split des données d'entraînement et de test :
    val splits = dfprocessed.randomSplit(Array(0.9, 0.1))
    val dftraining = splits(0)
    val dftest = splits(1)

    println("Split terminé")

    // Utiliser les données textuelles

    // La première étape est séparer les textes en mots (ou tokens)
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    // On veut retirer les stop words pour ne pas encombrer le modèle avec
    // des mots qui ne véhiculent pas de sens.
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("wordsremoved")

    // La partie TF de TF-IDF est faite avec la classe CountVectorizer.
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setInputCol("wordsremoved")
      .setOutputCol("tf")

    // Trouvez la partie IDF.
    // On veut écrire l’output de cette étape dans une colonne “tfidf”.
    val idf = new IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")

    // Assembler les features dans une seule colonne “features”.
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed","currency_indexed"))
      .setOutputCol("features")

    // Le modèle de classification
    // il s’agit d’une régression logistique qu'on définit de la façon suivante:
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.6, 0.4))
      .setTol(1.0e-6)
      .setMaxIter(300)

    // Mise en place d'un Pipeline qui chaîne tous les stages de transformation précédents :
    val pipeline2 = new Pipeline()
      .setStages(Array(tokenizer, remover, countVectorizer, idf, assembler,lr))

    // Définition de la grille de recherche pour tuning des hyper-paramètres :
    // On sélectionne le point de la grille où l’erreur de validation est la plus faible
    // et on garde les valeurs d’hyper-paramètres de ce point.
    val paramGrid = new ParamGridBuilder()
      .addGrid(countVectorizer.minDF, Array(55.0, 75.0, 95.0))
      .addGrid(lr.regParam, Array(10.0e-8, 10.0e-6, 10.0e-4, 10.0e-2))
      .build()

    // Evaluator : score f1
    val binEvaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    // pipeline de validation croisée :
    val cv = new CrossValidator()
      .setEstimator(pipeline2)
      .setEvaluator(binEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    // fit sur le jeu données training :
    val cvModel = cv.fit(dftraining)

    println("Cross-validation OK")

    // Pprédictions de chaque campagne :
    val df_WithPredictions = cvModel.transform(dftest)

    df_WithPredictions.select("project_id", "name","final_status", "predictions").show()

    // On affiche :
    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    val test_evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val f1Score = test_evaluator.evaluate(df_WithPredictions)

    println(f"Score F1 : $f1Score%.2f")

    df_WithPredictions.write.mode(SaveMode.Overwrite)
      .parquet("test_results")

    cvModel.write.overwrite().save("kickstarterModel")

  }
}
