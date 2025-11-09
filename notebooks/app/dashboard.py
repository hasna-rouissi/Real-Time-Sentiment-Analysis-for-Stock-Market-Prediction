import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Connexion à MongoDB
client = MongoClient("mongodb://mongodb:27017/")
db = client["stock_sentiment"]
collection_sgd = db["predicted_impact_sgd"]
collection_rf = db["predicted_impact_rf"]

# Chargement des données
data_sgd = pd.DataFrame(list(collection_sgd.find()))
data_rf = pd.DataFrame(list(collection_rf.find()))

# Nettoyage et normalisation
def clean_dataframe(df, label):
    if df.empty:
        st.error(f"❌ Le DataFrame '{label}' est vide.")
        st.stop()

    df.columns = [col.strip() for col in df.columns]

    required_columns = ['Date', 'Close', 'predicted_Close']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"❌ Colonne '{col}' manquante dans {label}")
            st.stop()

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['predicted_Close'] = pd.to_numeric(df['predicted_Close'], errors='coerce')

    df = df.dropna(subset=['Date', 'Close', 'predicted_Close'])
    df = df.sort_values('Date')

    return df

data_sgd = clean_dataframe(data_sgd, "SGD")
data_rf = clean_dataframe(data_rf, "RF")

# Vérification colonne Stock Name
if 'Stock Name' not in data_sgd.columns or 'Stock Name' not in data_rf.columns:
    st.error("❌ Colonne 'Stock Name' absente dans au moins une collection.")
    st.stop()

# Sidebar - Filtres
stock_list = sorted(set(data_sgd["Stock Name"].unique()) | set(data_rf["Stock Name"].unique()))
selected_stock = st.sidebar.selectbox("🏢 Choisir une entreprise", stock_list)

# Filtrage par entreprise
df_sgd = data_sgd[data_sgd["Stock Name"] == selected_stock]
df_rf = data_rf[data_rf["Stock Name"] == selected_stock]

# Filtre par année
years = sorted(df_sgd['Date'].dt.year.unique())
selected_years = st.sidebar.multiselect("📅 Filtrer par année", years, default=years)

df_sgd = df_sgd[df_sgd['Date'].dt.year.isin(selected_years)]
df_rf = df_rf[df_rf['Date'].dt.year.isin(selected_years)]

# Fusion des prédictions
merged = pd.merge(
    df_sgd[['Date', 'Close', 'predicted_Close']],
    df_rf[['Date', 'predicted_Close']],
    on='Date',
    suffixes=('_SGD', '_RF')
)

# Calcul des erreurs absolues
merged["Abs_Error_SGD"] = abs(merged["Close"] - merged["predicted_Close_SGD"])
merged["Abs_Error_RF"] = abs(merged["Close"] - merged["predicted_Close_RF"])

# Titre
st.title("📊 Comparaison des Modèles de Prédiction Boursière")

# Graphique réel vs prédictions
st.subheader("📈 Prix de Clôture : Réel vs Prédictions")
fig1 = px.line(
    merged, x='Date',
    y=['Close', 'predicted_Close_SGD', 'predicted_Close_RF'],
    labels={'value': 'Prix', 'Date': 'Date'},
    title=f"Prédictions - {selected_stock}"
)
st.plotly_chart(fig1)

# Graphique erreurs absolues
st.subheader("📉 Erreur Absolue")
fig2 = px.line(
    merged, x='Date',
    y=['Abs_Error_SGD', 'Abs_Error_RF'],
    labels={'value': 'Erreur absolue', 'Date': 'Date'},
    title=f"Erreur Absolue de Prédiction - {selected_stock}"
)
st.plotly_chart(fig2)

# Corrélation Sentiment vs Prix de Clôture
if "sentiment_score" in df_sgd.columns:
    st.subheader("🔄 Corrélation Sentiment vs Prix de Clôture")
    fig_corr = px.scatter(
        df_sgd, x="sentiment_score", y="Close",
        trendline="ols",
        title=f"Corrélation entre sentiment et prix - {selected_stock}"
    )
    st.plotly_chart(fig_corr)



# Sentiment du marché
if "sentiment_score" in df_sgd.columns:
    st.subheader("💬 Sentiment du Marché")
    fig3 = px.scatter(
        df_sgd, x='Date', y='sentiment_score',
        color='sentiment_score',
        size='Volume' if 'Volume' in df_sgd.columns else None,
        title=f"Score de Sentiment - {selected_stock}"
    )
    st.plotly_chart(fig3)


# Boxplot des prédictions
st.subheader("📦 Distribution des Prédictions")
melted = merged.melt(id_vars="Date", value_vars=["predicted_Close_SGD", "predicted_Close_RF"],
                     var_name="Modèle", value_name="Valeur")
fig5 = px.box(melted, x="Modèle", y="Valeur", title="Distribution des valeurs prédites")
st.plotly_chart(fig5)

# Tendance du sentiment
if "sentiment_score" in df_sgd.columns:
    st.subheader("📈 Tendance Moyenne du Sentiment")
    sentiment_evolution = df_sgd.groupby(pd.Grouper(key="Date", freq="W"))["sentiment_score"].mean().reset_index()
    fig6 = px.line(sentiment_evolution, x="Date", y="sentiment_score", title="Évolution hebdomadaire du sentiment")
    st.plotly_chart(fig6)

# Corrélation entre prédictions SGD et RF
st.subheader("📉 Corrélation entre prédictions SGD et RF")
correlation = merged["predicted_Close_SGD"].corr(merged["predicted_Close_RF"])
st.metric(label="Corrélation entre modèles", value=f"{correlation:.4f}")


# Footer
st.caption("📡 Données MongoDB — Collections : predicted_impact_sgd, predicted_impact_rf")
