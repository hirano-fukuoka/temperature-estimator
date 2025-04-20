import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

st.set_page_config(page_title="最終版：温度補正アプリ", layout="wide")
st.title("📊 β(t) + DTW + 補正式 + エラー耐性 + 列選択 対応アプリ")

st.markdown("内部温度（遅応答）を補正して、表面温度（速応答）を推定するアプリです。")

uploaded_file = st.file_uploader("📤 Excel または CSV ファイルをアップロード", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, header=None)
        else:
            df_raw = pd.read_excel(uploaded_file, header=None, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ ファイル読み込みエラー: {e}")
        st.stop()

    st.subheader("🔍 アップロード内容プレビュー（最初の10行）")
    st.dataframe(df_raw.head(10))

    # ===== ユーザーが列を指定 =====
    st.sidebar.header("📋 データ列を選択")
    columns = df_raw.columns.tolist()
    col_time = st.sidebar.selectbox("時間列", columns)
    col_internal = st.sidebar.selectbox("内部温度列", columns)
    col_surface = st.sidebar.selectbox("表面温度列", columns)

    # ===== 数値変換 & NaN削除 =====
    df = pd.DataFrame()
    df["time"] = pd.to_numeric(df_raw[col_time], errors="coerce")
    df["T_internal"] = pd.to_numeric(df_raw[col_internal], errors="coerce")
    df["T_surface"] = pd.to_numeric(df_raw[col_surface], errors="coerce")
    df.dropna(inplace=True)

    t = df["time"].values
    T_internal = df["T_internal"].values
    T_surface = df["T_surface"].values
    dt = np.mean(np.diff(t))

    # ===== β(t) 時間スパン補正 =====
    st.sidebar.header("⏳ β(t)補正設定")
    peak_center = st.sidebar.slider("ピーク中心 [s]", float(t[0]), float(t[-1]), float(t[len(t)//2]), step=0.1)
    peak_width = st.sidebar.slider("ピーク幅 [s]", 0.1, 20.0, 5.0)
    beta_base = st.sidebar.slider("ベースβ", 0.5, 3.0, 1.2)
    beta_peak = st.sidebar.slider("ピーク付近β", 0.1, 1.0, 0.6)

    def beta_func(t):
        return beta_peak + (beta_base - beta_peak) * np.exp(-((t - peak_center)**2) / (2 * peak_width**2))

    beta_vals = beta_func(t)
    t_scaled = np.cumsum(dt ** beta_vals)
    min_len = min(len(t_scaled), len(T_internal))
    T_internal = T_internal[:min_len]
    t_scaled = t_scaled[:min_len]
    t_trimmed = t[:min_len]

    try:
        interp_beta = interp1d(t_scaled, T_internal, kind="linear", fill_value="extrapolate", bounds_error=False)
        T_beta_scaled = interp_beta(t_trimmed)
    except Exception as e:
        st.error(f"補間エラー（βスパン）: {e}")
        st.stop()

    df = df.iloc[:len(T_beta_scaled)].copy()
    df["T_beta_scaled"] = T_beta_scaled

    # ===== DTW 整列 =====
    st.sidebar.header("🧠 DTW 整列")
    if st.sidebar.button("DTW補正を実行"):
        with st.spinner("DTW 処理中..."):
            T1 = df["T_beta_scaled"].astype(float).to_numpy().flatten()
            T2 = df["T_surface"].astype(float).to_numpy().flatten()
            mask = np.isfinite(T1) & np.isfinite(T2)
            T1_clean = T1[mask]
            T2_clean = T2[mask]

            try:
                distance, path = fastdtw(T1_clean, T2_clean, dist=euclidean)
                idx_i, idx_s = zip(*path)
                t_warped = df["time"].values[np.array(idx_s)]
                T_aligned = df["T_beta_scaled"].values[np.array(idx_i)]
                interp_dtw = interp1d(t_warped, T_aligned, kind="linear", fill_value="extrapolate", bounds_error=False)
                df["T_dtw_aligned"] = interp_dtw(df["time"])
            except Exception as e:
                st.error(f"DTWエラー: {e}")
                st.stop()

        st.success(f"✅ DTW補正完了（距離: {distance:.2f}）")

        # ===== 最適化 =====
        st.sidebar.header("📐 a×T + b×dT/dt + c 補正")
        if st.sidebar.button("推定式を最適化"):
            try:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)
                dTdt = np.gradient(df["T_dtw_aligned"], dt)

                def objective(params):
                    a, b, c = params
                    pred = a * df["T_dtw_aligned"] + b * dTdt + c
                    return np.mean((df["T_surface"] - pred) ** 2)

                res = minimize(objective, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
                a_opt, b_opt, c_opt = res.x

                df["T_predicted"] = a_opt * df["T_dtw_aligned"] + b_opt * dTdt + c_opt

                st.success("📌 最適化完了")
                st.info(f"a = {a_opt:.4f} / b = {b_opt:.4f} / c = {c_opt:.4f}")

                # ===== グラフ描画 =====
                st.subheader("📈 実測 vs 補正 vs 推定")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df["time"], df["T_surface"], label="実測（表面）")
                ax.plot(df["time"], df["T_dtw_aligned"], label="補正（内部）", linestyle=":")
                ax.plot(df["time"], df["T_predicted"], label="推定", linestyle="--")
                ax.set_xlabel("時間 [s]")
                ax.set_ylabel("温度 [℃]")
                ax.legend()
                st.pyplot(fig)

                # ===== 出力 =====
                st.download_button(
                    label="📥 結果をCSVでダウンロード",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="temperature_estimation_result.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"最適化処理エラー: {e}")
