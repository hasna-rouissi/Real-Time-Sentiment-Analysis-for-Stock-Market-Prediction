import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Chargement des données historiques
df = pd.read_csv("dataset_market_tweets_synthetic.csv")  # Remplace par ton chemin

# Nettoyage
required_cols = ["sentiment_score", "Open", "High", "Low", "Volume", "Close"]
df.dropna(subset=required_cols, inplace=True)

X = df[["sentiment_score", "Open", "High", "Low", "Volume"]].astype("float64")
y = df["Close"].astype("float64")

# Split pour entraînement
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

# Mise à l’échelle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Modèle SGD
model = SGDRegressor(max_iter=1000, tol=1e-3)
model.fit(X_train_scaled, y_train)

# Sauvegarde
joblib.dump(model, "sgd_regressor.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Modèle SGD entraîné et sauvegardé.")
