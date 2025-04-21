import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.stats import pearsonr
from dtw import dtw
from numpy.linalg import norm
import io

st.set_page_config(page_title="表面温度推定アプリ v11", layout="wide")
st.title("🌡 表面温度推定アプリ（v11｜手動/自動 最適化対応）")

uploaded_file = st.file_uploader("📤 CSV または Excel ファイルをアップロード", type=["csv", "xlsx"])
header_row = st.number_input("ヘッダーの行番号（0ベース）", min_value=0, value=0, step=1)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, header=header_row)
        else:
            df = pd.read_excel(uploaded_file, header=header_row, engine="openpyxl")
        df.columns = df.columns.astype(str)
    except Exception as e:
        st.error(f"❌ ファイル読み込みエラー: {e}")
        st.stop()

    st.success("✅ 読み込み成功")
    st.dataframe(df.astype(str))

    st.sidebar.header("📋 列選択")
    col_time = st.sidebar.selectbox("時間列", df.columns, index=0)
    col_internal = st.sidebar.selectbox("内部温度列", df.columns, index=1)
    col_surface = st.sidebar.selectbox("表面温度列", df.columns, index=2)

    try:
        time = pd.to_numeric(df[col_time], errors="coerce")
        T_internal = pd.to_numeric(df[col_internal], errors="coerce")
        T_surface = pd.to_numeric(df[col_surface], errors="coerce")
    except Exception as e:
        st.error(f"❌ 数値変換エラー: {e}")
        st.stop()

    st.sidebar.header("🛠 応答補正設定")
    dt = st.sidebar.number_input("サンプリング間隔 [s]", min_value=0.001, value=0.1, step=0.01)

    optimize_all = st.sidebar.checkbox("📌 スパン・シフトも含めて自動最適化", value=True)

    if optimize_all:
        st.sidebar.markdown("🔎 最小二乗法で最適化中...")

        def loss(params):
            a, b, c, scale, shift = params
            dTdt = np.gradient(T_internal, dt)
            T_pred = a * T_internal + b * dTdt + c
            t_scaled = (time + shift) * scale
            interp_func = interp1d(time, T_pred, bounds_error=False, fill_value="extrapolate")
            T_scaled = interp_func(t_scaled)
            mask = ~np.isnan(T_surface) & ~np.isnan(T_scaled)
            return np.mean((T_surface[mask] - T_scaled[mask])**2)

        result = minimize(loss, x0=[1.0, 0.0, 0.0, 1.0, 0.0], method="Nelder-Mead")
        a, b, c, scale, shift = result.x
    else:
        st.sidebar.markdown("✋ 手動パラメータ設定")
        a = st.sidebar.number_input("a（内部温度係数）", value=1.0)
        b = st.sidebar.number_input("b（傾き係数）", value=0.0)
        c = st.sidebar.number_input("c（オフセット）", value=0.0)
        scale = st.sidebar.slider("スケーリング倍率（スパン）", 0.1, 5.0, 1.0, step=0.1)
        shift = st.sidebar.slider("時間オフセット（シフト）[s]", -10.0, 10.0, 0.0, step=0.1)

    # 推定値算出
    dTdt = np.gradient(T_internal, dt)
    T_est = a * T_internal + b * dTdt + c
    t_scaled = (time + shift) * scale
    interp_func = interp1d(time, T_est, bounds_error=False, fill_value="extrapolate")
    T_est_scaled = interp_func(t_scaled)

    # マスクと指標計算
    mask = ~np.isnan(T_surface) & ~np.isnan(T_est_scaled)
    u = np.asarray(T_surface[mask]).flatten()
    v = np.asarray(T_est_scaled[mask]).flatten()
    r = np.corrcoef(u, v)[0, 1]
    rmse = np.sqrt(np.mean((u - v) ** 2))
    dtw_distance = dtw(u, v).normalizedDistance

    # グラフ
    st.subheader("📈 実測 vs 補正温度")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, T_surface, label="実測（表面）", linestyle="--", color="orange")
    ax.plot(time, T_est_scaled, label="推定（補正後）", color="blue")
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("温度 [℃]")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"### ✅ 使用パラメータ")
    st.code(f"a = {a:.4f}, b = {b:.4f}, c = {c:.4f}, scale = {scale:.4f}, shift = {shift:.4f}")
    st.markdown(f"**📏 相関係数**: {r:.4f}  **RMSE**: {rmse:.4f}  **DTW距離**: {dtw_distance:.4f}")

    # ダウンロード
    result_df = pd.DataFrame([{
        "a": a, "b": b, "c": c,
        "scale": scale, "shift": shift,
        "r": r, "rmse": rmse, "dtw": dtw_distance
    }])
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 結果CSVをダウンロード", csv, file_name="最適化結果.csv", mime="text/csv")
