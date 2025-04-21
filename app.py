import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from dtw import dtw
from numpy.linalg import norm

st.set_page_config(page_title="表面温度推定アプリ v9", layout="wide")
st.title("🌡 表面温度推定アプリ（v9｜自動係数最適化＋手動切替対応）")

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

    st.sidebar.header("⏳ 時間スケーリング補正")
    time_shift_scale = st.sidebar.slider("スケーリング倍率（スパン）", 0.1, 5.0, 1.0, step=0.1)
    time_shift_offset = st.sidebar.slider("時間オフセット（シフト）[s]", -10.0, 10.0, 0.0, step=0.1)

    mode = st.sidebar.radio("パラメータ設定モード", ["手動設定", "最小二乗法で自動最適化"])

    if mode == "手動設定":
        st.sidebar.header("📐 手動設定: 推定式係数")
        a = st.sidebar.number_input("係数 a（内部温度）", value=1.0)
        b = st.sidebar.number_input("係数 b（傾き）", value=0.0)
        c = st.sidebar.number_input("オフセット c", value=0.0)

        dTdt = np.gradient(T_internal, dt)
        T_estimated = a * T_internal + b * dTdt + c
        params_used = (a, b, c)

    else:
        st.sidebar.markdown("🔎 自動最適化中...")

        def loss(params):
            a, b, c = params
            dTdt = np.gradient(T_internal, dt)
            T_pred = a * T_internal + b * dTdt + c
            return np.mean((T_surface - T_pred) ** 2)

        result = minimize(loss, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
        a, b, c = result.x
        dTdt = np.gradient(T_internal, dt)
        T_estimated = a * T_internal + b * dTdt + c
        params_used = (a, b, c)
        st.sidebar.success(f"✅ 最適化完了: a={a:.3f}, b={b:.3f}, c={c:.3f}")

    # 時間スケーリング適用
    time_scaled = (time + time_shift_offset) * time_shift_scale
    interp_est = interp1d(time, T_estimated, bounds_error=False, fill_value="extrapolate")
    T_est_scaled = interp_est(time_scaled)

    # 実測と推定のDTW評価
    u_series = pd.to_numeric(df[col_surface], errors="coerce").dropna()
    v_series = pd.Series(T_est_scaled).dropna()
    min_len = min(len(u_series), len(v_series))
    u = u_series.to_numpy().flatten()[:min_len]
    v = v_series.to_numpy().flatten()[:min_len]
    distance = dtw(u, v).normalizedDistance

    # グラフ表示
    st.subheader("📈 実測 vs 補正温度")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, T_surface, label="実測（表面）", linestyle="--", color="orange")
    ax.plot(time, T_est_scaled, label="推定（補正後）", color="blue")
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("温度 [℃]")
    ax.legend()
    st.pyplot(fig)

    st.markdown(f"📌 使用パラメータ: `a = {params_used[0]:.3f}`, `b = {params_used[1]:.3f}`, `c = {params_used[2]:.3f}`")
    st.info(f"📏 DTW距離（正規化）: {distance:.4f}")
