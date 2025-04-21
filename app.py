import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.interpolate import interp1d

st.title("表面温度推定アプリ（完全安定版 v5）")

# ヘッダー指定 & ファイルアップロード
uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロードしてください", type=["csv", "xlsx"])
header_row = st.number_input("ヘッダーの行番号（0開始）", min_value=0, value=0, step=1)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, header=header_row)
        else:
            df = pd.read_excel(uploaded_file, header=header_row)
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        st.stop()

    df.columns = df.columns.astype(str)  # 列名が数値になるのを防ぐ

    st.success("データ読み込み成功")
    st.dataframe(df.head())

    # 列の選択
    col_time = st.selectbox("時間列を選択", df.columns, index=0)
    col_internal = st.selectbox("内部温度列を選択", df.columns, index=1)
    col_surface = st.selectbox("表面温度列を選択", df.columns, index=2)

    # 数値データを抽出
    try:
        time = pd.to_numeric(df[col_time], errors='coerce')
        T_internal = pd.to_numeric(df[col_internal], errors='coerce')
        T_surface = pd.to_numeric(df[col_surface], errors='coerce')
    except Exception as e:
        st.error(f"数値変換エラー: {e}")
        st.stop()

    # 応答補正設定
    st.sidebar.subheader("🛠 応答補正設定")
    sampling_interval = st.sidebar.number_input("サンプリング間隔 Δt [s]", min_value=0.001, value=0.1, step=0.01)
    time_shift = st.sidebar.number_input("時間シフト β [倍]", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

    # β(t) 時間圧縮：ピーク中心にスパン圧縮（簡易モデル）
    beta_t = 1 / (1 + np.exp(-(time - time.mean()))) * time_shift

    # 温度補正係数
    st.sidebar.subheader("📐 温度補正式係数")
    alpha = st.sidebar.number_input("温度係数 α", value=1.0)
    beta = st.sidebar.number_input("傾き係数 β (dT/dt)", value=0.0)
    offset = st.sidebar.number_input("オフセット c", value=0.0)

    # 補正実行
    try:
        dTdt = np.gradient(T_internal, sampling_interval)
        T_surface_estimated = alpha * T_internal + beta * dTdt + offset

        # 時間軸補正：補間 + スケーリング
        f_interp = interp1d(time, T_surface_estimated, bounds_error=False, fill_value="extrapolate")
        time_scaled = time * time_shift
        T_surface_estimated_scaled = f_interp(time_scaled)

        # DTWによる補正（速度ベース）
        u = T_surface.dropna().values.flatten()
        v = T_surface_estimated_scaled.dropna().values.flatten()

        if len(u) != len(v):
            min_len = min(len(u), len(v))
            u = u[:min_len]
            v = v[:min_len]

        dtw_result = dtw(u, v)
        dist = dtw_result.normalizedDistance

        # グラフ表示
        fig, ax = plt.subplots()
        ax.plot(time, T_surface, label="表面温度 (実測)", linestyle="--", color='orange')
        ax.plot(time, T_surface_estimated_scaled, label="表面温度 (推定)", color='blue')
        ax.set_xlabel("時間 [s]")
        ax.set_ylabel("温度 [℃]")
        ax.set_title("推定結果グラフ")
        ax.legend()
        st.pyplot(fig)

        st.info(f"📏 推定誤差 (DTW距離): {dist:.4f}")
    except Exception as e:
        st.error(f"補正または描画処理エラー: {e}")
