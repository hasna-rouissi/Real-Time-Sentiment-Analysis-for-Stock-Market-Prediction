from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, concat_ws
from pyspark.sql.types import StructType, StringType, FloatType
import pandas as pd
import joblib
from pymongo import MongoClient, UpdateOne

# Initialisation de Spark
spark = SparkSession.builder \
    .appName("SentimentImpactOnStock_SGD") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1") \
    .getOrCreate()

# Définition du schéma pour les messages Kafka
schema = StructType() \
    .add("Date", StringType()) \
    .add("Stock Name", StringType()) \
    .add("sentiment_score", FloatType()) \
    .add("Open", FloatType()) \
    .add("High", FloatType()) \
    .add("Low", FloatType()) \
    .add("Close", FloatType()) \
    .add("Volume", FloatType())

# Lecture des données depuis Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "stock_sentiment") \
    .option("startingOffsets", "latest") \
    .load()

# Conversion des données Kafka en DataFrame structuré
df_value = df_raw.selectExpr("CAST(value AS STRING)")
df_parsed = df_value.select(from_json(col("value"), schema).alias("data")).select("data.*")

# Chargement du modèle et du scaler
model = joblib.load("sgd_regressor.pkl")
scaler = joblib.load("scaler.pkl")

# Fonction appelée pour chaque batch
def write_and_predict(batch_df, batch_id):
    batch_df = batch_df.withColumn("_id", concat_ws("_", "Date", "Stock Name"))
    pdf = batch_df.toPandas()

    if pdf.empty:
        return

    print(f"📦 Batch {batch_id} reçu avec {len(pdf)} lignes")

     # Filtrer les colonnes nécessaires
    required_cols = ["sentiment_score", "Open", "High", "Low", "Volume"]
    pdf.dropna(subset=required_cols, inplace=True)

    if pdf.empty:
        print("🚫 Données insuffisantes après nettoyage.")
        return

    try:

    # Préparation des données pour la prédiction
        X = pdf[required_cols].astype("float64")
        X_scaled = scaler.transform(X)
        pdf["predicted_Close"] = model.predict(X_scaled)

    # Connexion à MongoDB et insertion
        client = MongoClient("mongodb://mongodb:27017/")
        db = client["stock_sentiment"]
        collection = db["predicted_impact_sgd"]

        operations = [
            UpdateOne({"_id": row["_id"]}, {"$set": row.to_dict()}, upsert=True)
            for _, row in pdf.iterrows()
        ]

        if operations:
            collection.bulk_write(operations)
            print(f"✅ Batch {batch_id} inséré dans MongoDB avec {len(operations)} prédictions")
        client.close()

    except Exception as e:
         
        print(f"❌ Erreur durant la prédiction ou MongoDB : {e}")

# Lancement du traitement de flux
query = df_parsed.writeStream \
    .trigger(processingTime="30 seconds") \
    .foreachBatch(write_and_predict) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoints/sgd_predict") \
    .start()

query.awaitTermination()
