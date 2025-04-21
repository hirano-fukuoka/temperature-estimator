import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from dtw import dtw
from numpy.linalg import norm

st.set_page_config(page_title="表面温度推定アプリ v7", layout="wide")
st.title("🌡 表面温度推定アプリ（完全安定版 v7）")

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

    st.sidebar.header("📐 推定式係数")
    alpha = st.sidebar.number_input("係数 α（内部温度）", value=1.0)
    beta = st.sidebar.number_input("係数 β（傾き）", value=0.0)
    offset = st.sidebar.number_input("オフセット c", value=0.0)

    st.sidebar.header("⏳ 時間スケーリング補正")
    time_shift_scale = st.sidebar.slider("スケーリング倍率（スパン）", 0.1, 5.0, 1.0, step=0.1)
    time_shift_offset = st.sidebar.slider("時間オフセット（シフト）[s]", -10.0, 10.0, 0.0, step=0.1)

    try:
        dTdt = np.gradient(T_internal, dt)
        T_estimated = alpha * T_internal + beta * dTdt + offset

        # 時間スケーリングとシフトの適用
        time_scaled = (time + time_shift_offset) * time_shift_scale
        interp_est = interp1d(time, T_estimated, bounds_error=False, fill_value="extrapolate")
        T_est_scaled = interp_est(time_scaled)

        # 実測と推定のDTW比較（要1D, NaN除去, 同長）
        u_series = pd.to_numeric(df[col_surface], errors="coerce").dropna()
        v_series = pd.Series(T_est_scaled).dropna()
        min_len = min(len(u_series), len(v_series))
        u = u_series.to_numpy().flatten()[:min_len]
        v = v_series.to_numpy().flatten()[:min_len]

        distance = dtw(u, v, dist=lambda x, y: norm(x - y)).normalizedDistance

        # グラフ表示
        st.subheader("📈 実測 vs 補正温度")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(time, T_surface, label="実測（表面）", linestyle="--", color="orange")
        ax.plot(time, T_est_scaled, label="補正推定", color="blue")
        ax.set_xlabel("時間 [s]")
        ax.set_ylabel("温度 [℃]")
        ax.legend()
        st.pyplot(fig)

        st.info(f"📏 DTW距離（正規化）: {distance:.4f}")

    except Exception as e:
        st.error(f"❌ 処理エラー: {type(e).__name__}: {e}")
