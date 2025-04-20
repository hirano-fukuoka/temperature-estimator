import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

st.set_page_config(page_title="時間スパン補正 + DTWアプリ", layout="wide")
st.title("🌡 β(t) 時間補正 + DTW による表面温度推定（エラー耐性強化）")

st.markdown("""
このアプリは以下の処理を行います：

- `β(t)` による時間軸の局所圧縮（スパン補正）
- `Dynamic Time Warping (DTW)` による時系列整列
- `a × T + b × dT/dt + c` の補正式による推定（最小誤差で最適化）
- **エラー耐性強化**（1Dチェック、NaN/Inf除去、安全補間）
""")

uploaded_file = st.file_uploader("📤 CSV または Excel ファイルをアップロード", type=["csv", "xlsx"])

if uploaded_file:
    # --- ファイル読み込み（安全）
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ 読み込みエラー: {e}")
        st.stop()

    st.success("✅ ファイル読み込み完了")
    st.dataframe(df.head())

    # --- 必須列チェック
    required = {"time", "T_internal", "T_surface"}
    if not required.issubset(df.columns):
        st.error(f"⛔ 以下の列が必要です: {required}")
        st.stop()

    df.dropna(subset=["time", "T_internal", "T_surface"], inplace=True)
    t = df["time"].values
    T_internal = df["T_internal"].values
    T_surface = df["T_surface"].values
    dt = np.mean(np.diff(t))

    # === β(t) 時間スパン補正 ===
    st.sidebar.header("⏳ β(t) スパン補正設定")
    peak_center = st.sidebar.slider("ピーク中心 [s]", float(t[0]), float(t[-1]), float(t[len(t)//2]), step=0.1)
    peak_width = st.sidebar.slider("ピーク幅 [s]", 0.1, 20.0, 5.0, step=0.1)
    beta_base = st.sidebar.slider("ベースβ", 0.5, 3.0, 1.2, step=0.1)
    beta_peak = st.sidebar.slider("ピーク付近β", 0.1, 1.0, 0.6, step=0.05)

    def beta_func(t):
        return beta_peak + (beta_base - beta_peak) * np.exp(-((t - peak_center)**2) / (2 * peak_width**2))

    beta_vals = beta_func(t)
    dt_beta = dt ** beta_vals
    t_scaled = np.cumsum(dt_beta)

    min_len = min(len(t_scaled), len(T_internal))
    t_scaled = t_scaled[:min_len]
    T_internal = T_internal[:min_len]
    t_trimmed = t[:min_len]

    try:
        interp_beta = interp1d(t_scaled, T_internal, kind="linear", fill_value="extrapolate", bounds_error=False)
        T_beta_scaled = interp_beta(t_trimmed)
    except Exception as e:
        st.error(f"補間エラー（βスケーリング）: {e}")
        st.stop()

    df = df.iloc[:len(T_beta_scaled)].copy()
    df["T_beta_scaled"] = T_beta_scaled

    # === DTW実行 ===
    st.sidebar.header("🧠 DTW 時系列整列")
    run_dtw = st.sidebar.button("DTW補正を実行")

    if run_dtw:
        with st.spinner("DTW 整列中..."):
            T1 = df["T_beta_scaled"].values.astype(float)
            T2 = df["T_surface"].values.astype(float)
            mask = ~np.isnan(T1) & ~np.isnan(T2) & np.isfinite(T1) & np.isfinite(T2)
            T1_clean = T1[mask]
            T2_clean = T2[mask]

            try:
                distance, path = fastdtw(T1_clean, T2_clean, dist=euclidean)
                idx_i, idx_s = zip(*path)

                t_surface_warped = df["time"].values[np.array(idx_s)]
                T_internal_warped = df["T_beta_scaled"].values[np.array(idx_i)]

                interp_dtw = interp1d(
                    t_surface_warped, T_internal_warped, kind="linear",
                    fill_value="extrapolate", bounds_error=False
                )
                df["T_dtw_aligned"] = interp_dtw(df["time"])
            except Exception as e:
                st.error(f"DTW処理エラー: {e}")
                st.stop()

        st.success(f"✅ DTW補正完了（距離: {distance:.2f}）")

        # === 自動最適化 ===
        st.sidebar.header("📐 a × T + b × dT/dt + c 推定")
        run_fit = st.sidebar.button("係数最適化を実行")

        if run_fit:
            try:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.dropna(inplace=True)

                dTdt = np.gradient(df["T_dtw_aligned"], dt)

                def objective(params):
                    a, b, c = params
                    pred = a * df["T_dtw_aligned"] + b * dTdt + c
                    return np.mean((df["T_surface"] - pred)**2)

                res = minimize(objective, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
                a_opt, b_opt, c_opt = res.x

                df["T_predicted"] = a_opt * df["T_dtw_aligned"] + b_opt * dTdt + c_opt

                st.success("✅ 最適化完了")
                st.info(f"📌 最適係数：a = {a_opt:.4f}、b = {b_opt:.4f}、c = {c_opt:.4f}")

                # === グラフ比較
                st.subheader("📈 実測 vs 補正 vs 推定")
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df["time"], df["T_surface"], label="実測（表面）")
                ax.plot(df["time"], df["T_dtw_aligned"], label="内部温度（DTW補正）", linestyle=":")
                ax.plot(df["time"], df["T_predicted"], label="推定温度", linestyle="--")
                ax.set_xlabel("時間 [s]")
                ax.set_ylabel("温度 [℃]")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # === CSV出力
                st.download_button(
                    label="📥 CSVでダウンロード",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="corrected_temperature_result.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"最適化処理中のエラー: {e}")
