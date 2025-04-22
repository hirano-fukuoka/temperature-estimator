import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import prepare_train_data, prepare_predict_data
from sklearn.ensemble import RandomForestRegressor

st.title("T_surface 予測アプリ")

# STEP 1: 学習データアップロード
st.header("1. 学習用CSVをアップロード")
train_file = st.file_uploader("学習用データ（T_internal, T_surfaceを含む）", type="csv", key="train")

model = None
if train_file:
    train_df = pd.read_csv(train_file)
    X_train, y_train = prepare_train_data(train_df)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.success("学習完了 ✅")

# STEP 2: 予測データアップロード
st.header("2. 予測対象のCSVをアップロード")
test_file = st.file_uploader("予測用データ（T_internalのみでもOK）", type="csv", key="test")

if model and test_file:
    test_df = pd.read_csv(test_file)
    X_test, test_df_with_lags = prepare_predict_data(test_df)
    y_pred = model.predict(X_test)

    # 予測結果をデータフレームに追加
    test_df_with_lags["Predicted_T_surface"] = y_pred

    # 可視化
    st.subheader("予測結果グラフ")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_df_with_lags["Predicted_T_surface"], label="予測T_surface")
    ax.set_xlabel("タイムステップ")
    ax.set_ylabel("温度 [°C]")
    ax.legend()
    st.pyplot(fig)

    # CSV出力
    st.subheader("予測結果のダウンロード")
    output_csv = test_df_with_lags[["time", "Predicted_T_surface"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="予測結果CSVをダウンロード",
        data=output_csv,
        file_name="predicted_t_surface.csv",
        mime="text/csv"
    )
