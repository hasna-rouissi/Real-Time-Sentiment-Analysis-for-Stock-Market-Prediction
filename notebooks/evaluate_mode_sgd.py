from pymongo import MongoClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def main():
    # Connexion à MongoDB
    client = MongoClient("mongodb://mongodb:27017/")
    db = client["stock_sentiment"]
    collection = db["predicted_impact_sgd"]

    # Chargement des données
    print("Chargement des prédictions depuis MongoDB...")
    data = pd.DataFrame(list(collection.find()))

    if data.empty:
        print("Aucune donnée trouvée dans la collection.")
        return

    # Vérification des colonnes nécessaires
    if "Close" not in data.columns or "predicted_Close" not in data.columns:
        print("Les colonnes 'Close' et 'predicted_Close' sont manquantes.")
        return

    # Nettoyage des données
    data = data[["Close", "predicted_Close"]].dropna()

    if data.empty:
        print("Pas de données valides pour évaluation après suppression des valeurs manquantes.")
        return

    # Calcul des métriques
    mae = mean_absolute_error(data["Close"], data["predicted_Close"])
    rmse = np.sqrt(mean_squared_error(data["Close"], data["predicted_Close"]))

    print("Évaluation du modèle :")
    print(f"MAE  (Mean Absolute Error) : {mae}")
    print(f"RMSE (Root Mean Squared Error) : {rmse}")

if __name__ == "__main__":
    main()