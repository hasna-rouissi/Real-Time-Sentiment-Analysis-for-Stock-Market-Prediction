import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Charger tes données historiques (ex: depuis CSV ou Mongo)
df = pd.read_csv("dataset_market_tweets_synthetic.csv")  # ou lire depuis MongoDB

required_cols = ["sentiment_score", "Open", "High", "Low", "Volume", "Close"]
df.dropna(subset=required_cols, inplace=True)

X = df[["sentiment_score", "Open", "High", "Low", "Volume"]].astype('float64')
y = df["Close"].astype('float64')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraînement du modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sauvegarde
joblib.dump(model, "model_rf.pkl")
print("✅ Modèle entraîné et sauvegardé dans 'model_rf.pkl'")
