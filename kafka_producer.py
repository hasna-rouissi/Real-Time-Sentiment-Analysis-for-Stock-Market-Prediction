from kafka import KafkaProducer
import pandas as pd
import json
import time
import os

import socket

def wait_for_kafka(host, port, timeout=60):
    print(f"Attente de Kafka sur {host}:{port} ...")
    start = time.time()
    while True:
        try:
            socket.create_connection((host, port), timeout=2)
            print("Kafka est disponible.")
            return
        except OSError:
            time.sleep(2)
            if time.time() - start > timeout:
                raise TimeoutError(f"Kafka sur {host}:{port} n'est pas disponible après {timeout} secondes.")

wait_for_kafka('kafka', 29092)
time.sleep(10)

csv_path = os.path.join("data", "dataset_market_tweets_synthetic.csv")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"Fichier non trouvé : {csv_path}")
    exit(1)


producer = KafkaProducer(
    bootstrap_servers='kafka:29092',  
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = "stock_sentiment"


for i, row in df.iterrows():
    try:
        message = row.to_dict()
        producer.send(topic, value=message)
        print(f"Sent ({i+1}/{len(df)}): {message}")
        time.sleep(0.05) 
    except Exception as e:
        print(f" Erreur à la ligne {i}: {e}")


producer.flush()
producer.close()
print("Tous les messages ont été envoyés.")
