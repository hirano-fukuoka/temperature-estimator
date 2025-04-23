import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ======= モデル処理（ラグ生成 + 学習用/予測用データ作成） =======

def create_lag_features(df, n_lags=20):
    for lag in range(1, n_lags+1):
        df[f"T_internal_lag{lag}"] = df["T_internal"].shift(lag)
    return df.dropna()

def prepare_train_data(df, n_lags=20):
    df_lag = create_lag_features(df.copy(), n_lags)
    X = df_lag[[f"T_internal_lag{i}" for i in range(1, n_lags+1)]]
    y = df_lag["T_surface"]
    return X, y

def prepare_predict_data(df, n_lags=20):
    df_lag = create_lag_features(df.copy(), n_lags)
    X = df_lag[[f"T_internal_lag{i}" for i in range(1, n_lags+1)]]
    return X, df_lag

# ======= Streamlit UI =======

st.set_page_config(page_title="T_surface 予測", layout="wide")
st.title("🌡️ T_surface 予測アプリ（熱電対 → ファイバー型温度センサー）")

# 1. 学習用データのアップロード
st.header("1️⃣ 学習データをアップロード（T_internal, T_surface を含む）")
train_file = st.file_uploader("学習用CSVファイル", type="csv", key="train")

model = None
if train_file:
    train_df = pd.read_csv(train_file)
    if "T_internal" in train_df.columns and "T_surface" in train_df.columns:
        X_train, y_train = prepare_train_data(train_df)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success("✅ 学習が完了しました")
    else:
        st.error("T_internal と T_surface の両方の列が必要です")

# 2. 予測対象データのアップロード
st.header("2️⃣ 予測用データをアップロード（T_internal 必須）")
test_file = st.file_uploader("予測対象のCSVファイル", type="csv", key="test")

if model and test_file:
    test_df = pd.read_csv(test_file)
    if "T_internal" not in test_df.columns:
        st.error("T_internal 列が予測データに存在しません")
    else:
        X_test, df_lagged = prepare_predict_data(test_df)
        y_pred = model.predict(X_test)

        # 予測結果を追加
        df_lagged["Predicted_T_surface"] = y_pred

        # グラフ描画
        st.subheader("📊 グラフ：T_internal・T_surface・予測T_surface")
        fig, ax = plt.subplots(figsize=(12, 5))

        time = df_lagged["time"] if "time" in df_lagged.columns else df_lagged.index

        ax.plot(time, df_lagged["T_internal"], label="T_internal (内部温度)", color="tab:blue")
        if "T_surface" in df_lagged.columns:
            ax.plot(time, df_lagged["T_surface"], label="T_surface (実測)", color="tab:green")
        ax.plot(time, df_lagged["Predicted_T_surface"], label="T_surface (予測)", color="tab:red", linestyle="--")

        ax.set_xlabel("時間 [s]")
        ax.set_ylabel("温度 [°C]")
        ax.legend()
        ax.set_title("内部温度と予測・実測の表面温度比較")
        st.pyplot(fig)

        # 予測結果CSV出力
        st.subheader("💾 予測結果CSVのダウンロード")
        output_csv = df_lagged[["time", "Predicted_T_surface"]].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="予測結果CSVをダウンロード",
            data=output_csv,
            file_name="predicted_t_surface.csv",
            mime="text/csv"
        )
