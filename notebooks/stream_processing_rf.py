from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, concat_ws
from pyspark.sql.types import StructType, StringType, FloatType
from joblib import load
from pymongo import MongoClient, UpdateOne
import pandas as pd
import os

# === 1. Initialisation Spark ===
spark = SparkSession.builder \
    .appName("SentimentImpactOnStock_RF_Predict") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1") \
    .getOrCreate()

# === 2. Schéma des messages Kafka ===
schema = StructType() \
    .add("Date", StringType()) \
    .add("Stock Name", StringType()) \
    .add("sentiment_score", FloatType()) \
    .add("Open", FloatType()) \
    .add("High", FloatType()) \
    .add("Low", FloatType()) \
    .add("Close", FloatType()) \
    .add("Volume", FloatType())

# === 3. Chargement du modèle pré-entraîné ===
model_path = "model_rf.pkl"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Modèle non trouvé à : {model_path}")
model = load(model_path)

# === 4. Lecture du stream Kafka ===
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:29092") \
    .option("subscribe", "stock_sentiment") \
    .option("startingOffsets", "latest") \
    .load()

df_value = df_raw.selectExpr("CAST(value AS STRING)")
df_parsed = df_value.select(from_json(col("value"), schema).alias("data")).select("data.*")

# === 5. Prédiction sur chaque micro-batch ===
def write_and_predict_only(batch_df, batch_id):
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
        # Prédiction
        X = pdf[required_cols].astype("float64")
        pdf["predicted_Close"] = model.predict(X)

        # Insertion MongoDB
        client = MongoClient("mongodb://mongodb:27017/")
        db = client["stock_sentiment"]
        collection = db["predicted_impact_rf"]

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

# === 6. Lancement du stream Kafka → Spark → MongoDB ===
query = df_parsed.writeStream \
    .trigger(processingTime="30 seconds") \
    .foreachBatch(write_and_predict_only) \
    .outputMode("append") \
    .option("checkpointLocation", "/tmp/checkpoints/rf_predict") \
    .start()

query.awaitTermination()
