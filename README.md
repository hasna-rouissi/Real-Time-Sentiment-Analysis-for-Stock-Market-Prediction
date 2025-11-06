# Real-Time-Sentiment-Analysis-for-Stock-Market-Prediction
This project aims to develop a real-time predictive system that forecasts stock market prices by analyzing sentiment trends from social media, particularly tweets, combined with historical financial data. It leverages Big Data technologies and Machine Learning to process, model, and visualize market dynamics in a distributed environment.


---

##  Technologies Used

| Component | Technology |
|------------|-------------|
| Data Streaming | **Apache Kafka** |
| Big Data Processing | **Apache Spark** |
| Machine Learning | **Scikit-learn (RandomForest, SGDRegressor)** |
| Data Storage | **MongoDB** |
| Visualization | **Streamlit Dashboard** |
| Language | **Python** |
| Containerization | **Docker / Docker Compose** |

---

## Model Training and Evaluation

Two models were trained on a combination of **sentiment-labeled tweets** and **financial time-series data**:
- **SGDRegressor** – fast online learning for streaming data  
- **RandomForestRegressor** – ensemble-based, robust against volatility  

**Evaluation Metrics:**
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)

**Results:**  
The **Random Forest model** outperformed the linear model, providing better accuracy and stability during rapid market changes.

---

##  Dashboard Insights (Streamlit)

The dashboard allows users to:
- Visualize **real-time predictions** of stock prices  
- Analyze **positive, negative, and neutral sentiments** from tweets  
- Compare **actual vs. predicted values**  
- Explore **sentiment impact** on market volatility  

---

##  Future Improvements

✅ Integrate live Twitter API streaming (instead of CSV-based simulation)  
✅ Extend to other financial assets (cryptocurrency, commodities, etc.)  
✅ Fine-tune deep learning models (e.g., LSTM, BERT, or Phi-2)  
✅ Deploy on cloud infrastructure for scalability  

---

