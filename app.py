import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="金型温度推定アプリ", layout="wide")
st.title("🌡️ 金型内部温度から表面温度を推定するアプリ")

# --- ファイルアップロード ---
uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロードしてください", type=["csv", "xlsx"])

if uploaded_file:
    # --- ファイルの読み込み ---
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("アップロードされたデータ")
    st.write(df.head())

    required_columns = {"time", "T_internal", "T_surface"}
    if not required_columns.issubset(df.columns):
        st.error(f"列名に {required_columns} を含めてください")
        st.stop()

    # --- パラメータ設定 ---
    tau = st.slider("熱電対の応答遅れ τ（秒）", 1.0, 10.0, 5.0)
    dt = st.slider("サンプリング周期 Δt（秒）", 0.5, 5.0, 1.0)
    alpha = dt / (tau + dt)

    # --- 応答補正関数（逆1次遅れ） ---
    def correct_response(measured, alpha):
        estimated = [measured.iloc[0]]
        for t in range(1, len(measured)):
            T_est = (measured.iloc[t] - (1 - alpha) * measured.iloc[t - 1]) / alpha
            estimated.append(T_est)
        return estimated

    # --- 応答補正処理 ---
    df["T_internal_corrected"] = correct_response(df["T_internal"], alpha)

    # --- 線形回帰モデルで推定 ---
    model = LinearRegression()
    model.fit(df[["T_internal_corrected"]], df["T_surface"])
    df["T_surface_predicted"] = model.predict(df[["T_internal_corrected"]])

    st.subheader("📈 推定結果プロット")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"], df["T_surface"], label="実測（表面）", linewidth=2)
    ax.plot(df["time"], df["T_surface_predicted"], label="推定（補正後）", linestyle="--")
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("温度 [°C]")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 推定データの一部")
    st.dataframe(df[["time", "T_internal", "T_internal_corrected", "T_surface", "T_surface_predicted"]].head())
