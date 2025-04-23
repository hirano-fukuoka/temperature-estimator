import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ======= 共通処理 =======

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
st.title("🌡️ T_surface 予測アプリ（複数学習データ対応 + 評価付き）")

# === 学習用データアップロード ===
st.header("1️⃣ 学習用CSVをアップロード（複数可）")
train_files = st.file_uploader("T_internal, T_surface を含むCSVを複数選択", type="csv", accept_multiple_files=True)

model = None
combined_df = pd.DataFrame()

if train_files:
    dfs = []
    for idx, f in enumerate(train_files):
        df = pd.read_csv(f)
        if "T_internal" in df.columns and "T_surface" in df.columns:
            df["source_id"] = f.name
            dfs.append(df)
        else:
            st.warning(f"{f.name} に必要な列が見つかりません。スキップします。")
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        X_train, y_train = prepare_train_data(combined_df)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        st.success(f"✅ 学習完了: {len(dfs)}ファイル、{len(X_train)}件のサンプル")
    else:
        st.error("⚠️ 有効な学習用データが見つかりませんでした。")

# === 予測用データアップロード ===
st.header("2️⃣ 予測用CSVをアップロード（複数可）")
test_files = st.file_uploader("T_internal を含むCSVを複数選択", type="csv", accept_multiple_files=True, key="test")

if model and test_files:
    all_results = []
    st.subheader("📈 予測結果グラフと評価")
    for idx, f in enumerate(test_files):
        df_test = pd.read_csv(f)
        if "T_internal" not in df_test.columns:
            st.warning(f"{f.name} に T_internal 列がありません。スキップします。")
            continue
        
        source_id = f.name
        X_test, df_lagged = prepare_predict_data(df_test)
        y_pred = model.predict(X_test)
        df_lagged["Predicted_T_surface"] = y_pred
        df_lagged["source_id"] = source_id

        # MSE 計算（実測がある場合のみ）
        mse = None
        if "T_surface" in df_lagged.columns:
            mse = mean_squared_error(df_lagged["T_surface"], y_pred)
            st.markdown(f"**📁 {source_id} - MSE: `{mse:.4f}`**")

        # グラフ描画
        fig, ax = plt.subplots(figsize=(10, 4))
        time = df_lagged["time"] if "time" in df_lagged.columns else df_lagged.index

        ax.plot(time, df_lagged["T_internal"], label="T_internal", color="tab:blue")
        if "T_surface" in df_lagged.columns:
            ax.plot(time, df_lagged["T_surface"], label="T_surface (実測)", color="tab:green")
        ax.plot(time, df_lagged["Predicted_T_surface"], label="T_surface (予測)", color="tab:red", linestyle="--")

        ax.set_title(f"{source_id}")
        ax.set_xlabel("時間 [s]")
        ax.set_ylabel("温度 [°C]")
        ax.legend()
        st.pyplot(fig)

        all_results.append(df_lagged)

    # === 全予測CSVの結合と出力 ===
    st.subheader("💾 全ファイルの予測結果CSVをダウンロード")
    output_df = pd.concat(all_results, ignore_index=True)
    csv = output_df[["source_id", "time", "T_internal", "Predicted_T_surface"] + (["T_surface"] if "T_surface" in output_df.columns else [])]
    csv_bytes = csv.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 予測結果CSVをダウンロード",
        data=csv_bytes,
        file_name="predicted_results.csv",
        mime="text/csv"
    )
