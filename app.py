import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
import tensorflow as tf
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Tesla Deep Learning Forecaster", layout="wide", page_icon="⚡")

st.title("⚡ Tesla Stock Prediction Portal")
st.markdown("This dashboard leverages advanced LSTM Networks combined with TA-Lib technical indicators to forecast TSLA stock horizons.")

# --- Sidebar ---
st.sidebar.header("Data & Model Config")

# 1. File Uploader
uploaded_file = st.sidebar.file_uploader("Upload TSLA.csv", type=["csv"])

# 2. Sidebar Parameters
sel_model = st.sidebar.selectbox("Neural Network Architecture", ['SimpleRNN', 'LSTM', 'StackedLSTM', 'TunedLSTM'])
sel_horizon = st.sidebar.selectbox("Prediction Horizon Options", [1, 5, 10], format_func=lambda x: f"{x}-Day Ahead")
lookback_val = st.sidebar.slider("Lookback Window (Sequence Length)", min_value=30, max_value=90, value=60, step=10)

predict_ahead_btn = st.sidebar.button("Predict Unknown Future! 🚀")

# --- Core Logic ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    
    st.subheader("1. Raw Data Integrity")
    st.dataframe(df.tail(3))
    
    # Preprocessing
    df.drop_duplicates(inplace=True)
    df.ffill(inplace=True)
    
    # Feature Engineering (Identical Pipeline to Rubric)
    try:
        close_ser = df['Adj Close']
        df['RSI_14'] = ta.momentum.RSIIndicator(close=close_ser, window=14).rsi()
        df['MACD'] = ta.trend.MACD(close=close_ser).macd()
        
        bb = ta.volatility.BollingerBands(close=close_ser, window=20)
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        
        df['SMA_7'] = ta.trend.SMAIndicator(close=close_ser, window=7).sma_indicator()
        df['SMA_21'] = ta.trend.SMAIndicator(close=close_ser, window=21).sma_indicator()
        df.dropna(inplace=True)
    except Exception as e:
        st.error(f"Cannot parse technical indicators from CSV: {e}")
        st.stop()
        
    feat_matrix = df.values
    targ_matrix = df[['Adj Close']].values
    
    # Native Scaling
    f_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_f = f_scaler.fit_transform(feat_matrix)
    
    t_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_t = t_scaler.fit_transform(targ_matrix)
    
    # Enforce shape requirements
    if len(df) < lookback_val + sel_horizon:
        st.warning("Insufficient rows to fulfill the lookback window constraints.")
        st.stop()
        
    model_path = f"models/model_{sel_model}_{sel_horizon}d.h5"
    
    try:
        # Load pre-trained execution from Colab. 
        # compile=False avoids custom metric load errors.
        model = load_model(model_path, compile=False)
    except OSError:
        st.error(f"❌ Model `{model_path}` Missing! Train models heavily in your Colab Jupyter Notebook and ensure the `models/` directory natively exists beside this app.")
        st.stop()

    def generate_seqs(x_data, y_data, w_size, h_size):
        X, y = [], []
        # Simulate identical sequencing to properly calculate testing RMSE
        for i in range(len(x_data) - w_size - h_size + 1):
            X.append(x_data[i : i + w_size])
            y.append(y_data[i + w_size + h_size - 1])
        return np.array(X), np.array(y)
    
    X_test, y_test_s = generate_seqs(scaled_f, scaled_t, lookback_val, sel_horizon)
    
    # -----------------
    # INFERENCE & EVAL
    # -----------------
    y_pred_s = model.predict(X_test, verbose=0)
    
    preds_usd = t_scaler.inverse_transform(y_pred_s).flatten()
    actuals_usd = t_scaler.inverse_transform(y_test_s.reshape(-1, 1)).flatten()
    
    # Align valid plot dates
    valid_plot_dates = df.index[lookback_val + sel_horizon - 1 :]
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(actuals_usd, preds_usd))
    mae = mean_absolute_error(actuals_usd, preds_usd)
    r2 = r2_score(actuals_usd, preds_usd)
    
    # Metrics Table Layout
    st.markdown("---")
    st.subheader(f"2. Validation Metrics ({sel_model} @ {sel_horizon}D Horizon)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Root Mean Squared Error", f"${rmse:.2f}", delta_color="inverse")
    col2.metric("Mean Absolute Error", f"${mae:.2f}")
    col3.metric("R² Variance Score", f"{r2:.4f}")
    
    # Interactive Plotly Chart
    st.markdown("---")
    st.subheader("3. Forecasting Overlay (Actual vs Predictions)")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid_plot_dates, y=actuals_usd, name="Real Market Price", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=valid_plot_dates, y=preds_usd, name="AI Prediction", line=dict(color="red", dash="dot")))
    fig.update_layout(xaxis_title="Date", yaxis_title="Adjusted Close", hovermode="x unified", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. Unknown Extrapolation "Predict Next N Days" button
    if predict_ahead_btn:
        st.markdown("---")
        st.subheader("🔮 Unknown Future Extrapolation")
        
        # Grab strictly the most recent window of known data
        recent_chunk = scaled_f[-lookback_val:]
        recent_chunk_expanded = recent_chunk.reshape(1, lookback_val, scaled_f.shape[1])
        
        future_pred_s = model.predict(recent_chunk_expanded, verbose=0)
        future_val_usd = t_scaler.inverse_transform(future_pred_s).flatten()[0]
        
        future_date = df.index[-1] + pd.Timedelta(days=sel_horizon)
        
        st.success(f"Based on the last consecutive {lookback_val} days of momentum, the **{sel_model}** projects the price on **{future_date.strftime('%A, %B %d, %Y')}** to hit:")
        st.markdown(f"<h1 style='text-align: center; color: green;'>${future_val_usd:,.2f}</h1>", unsafe_allow_html=True)
        
else:
    st.info("👋 Welcome! Please upload your dataset using the File Uploader in the sidebar to securely initialize the dashboard.")
