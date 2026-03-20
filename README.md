🚀 Tesla Stock Price Prediction using Deep Learning

Predicting Tesla stock prices using SimpleRNN and LSTM models with an interactive Streamlit dashboard.

📌 Project Overview

This project focuses on building a deep learning-based time series model to predict Tesla stock closing prices using historical data.

We implemented and compared:

🔹 Simple Recurrent Neural Network (SimpleRNN)

🔹 Long Short-Term Memory (LSTM)

The project also includes a Streamlit web app for real-time predictions and visualization.

🎯 Objectives

Predict Tesla stock closing prices

Compare performance of RNN vs LSTM

Build an end-to-end ML pipeline

Deploy predictions using a user-friendly dashboard

🧠 Tech Stack

Programming Language: Python

Libraries:

pandas, numpy

matplotlib

scikit-learn

tensorflow / keras

Deployment/UI: Streamlit

📂 Project Structure
    ├── app.py                       # Streamlit dashboard
    ├── tesla_stock_prediction.ipynb # Full ML pipeline
    ├── TSLA.csv                     # Dataset
    └── README.md                    # Project documentation

▶️ How to Run the Project
    1️⃣ Clone the Repository
          git clone https://github.com/your-username/tesla-stock-prediction.git
          cd tesla-stock-prediction
    2️⃣ Install Dependencies
          pip install -r requirements.txt
    3️⃣ Run Jupyter Notebook
          jupyter notebook tesla_stock_prediction.ipynb
    4️⃣ Run Streamlit App
          streamlit run app.py
