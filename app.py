import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import load_and_prepare_data, train_model
from sklearn.metrics import mean_squared_error

st.title("T_surface 予測アプリ（熱電対 → ファイバー型センサー）")

uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file:
    X, y, df = load_and_prepare_data(uploaded_file)
    model = train_model(X, y)
    y_pred = model.predict(X)

    st.subheader("予測 vs 実測")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y.values, label="実測T_surface")
    ax.plot(y_pred, label="予測T_surface")
    ax.set_xlabel("タイムステップ")
    ax.set_ylabel("温度 [°C]")
    ax.legend()
    st.pyplot(fig)

    mse = mean_squared_error(y, y_pred)
    st.write(f"平均二乗誤差（MSE）: {mse:.3f}")
