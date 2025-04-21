import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

st.set_page_config(page_title="完全安定版 v4 温度補正アプリ", layout="wide")
st.title("🌡 完全安定版 v4：DTW 1D保証 & Arrow完全対策")

uploaded_file = st.file_uploader("📤 CSVまたはExcelファイルをアップロード", type=["csv", "xlsx"])

if uploaded_file:
    st.sidebar.header("🗂 ヘッダー行の指定")
    header_row = st.sidebar.number_input("ヘッダーの行番号（最初が0）", min_value=0, max_value=50, value=0, step=1)

    try:
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file, header=header_row)
        else:
            df_raw = pd.read_excel(uploaded_file, header=header_row, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ ファイル読み込みエラー: {e}")
        st.stop()

    df_raw = df_raw.convert_dtypes()

    st.subheader("🔍 アップロード内容")
    try:
        st.dataframe(df_raw.astype(str))
    except Exception as e:
        st.warning("⚠️ 表示に失敗しました。代替表示を行います。")
        st.write(df_raw.to_string())

    st.sidebar.header("📋 列の選択")
    cols = df_raw.columns.tolist()
    col_time = st.sidebar.selectbox("時間列", cols)
    col_internal = st.sidebar.selectbox("内部温度列", cols)
    col_surface = st.sidebar.selectbox("表面温度列", cols)

    df = pd.DataFrame()
    df["time"] = pd.to_numeric(df_raw[col_time], errors="coerce").astype("float64")
    df["T_internal"] = pd.to_numeric(df_raw[col_internal], errors="coerce").astype("float64")
    df["T_surface"] = pd.to_numeric(df_raw[col_surface], errors="coerce").astype("float64")
    df.dropna(inplace=True)

    t = df["time"].values
    T_internal = df["T_internal"].values
    T_surface = df["T_surface"].values
    dt = np.mean(np.diff(t))

    st.sidebar.header("⏳ β(t)補正設定")
    peak_center = st.sidebar.slider("ピーク中心 [s]", float(t[0]), float(t[-1]), float(t[len(t)//2]), step=0.1)
    peak_width = st.sidebar.slider("ピーク幅 [s]", 0.1, 20.0, 5.0)
    beta_base = st.sidebar.slider("ベースβ", 0.5, 3.0, 1.2)
    beta_peak = st.sidebar.slider("ピークβ", 0.1, 1.0, 0.6)

    def beta_func(t): return beta_peak + (beta_base - beta_peak) * np.exp(-((t - peak_center)**2)/(2 * peak_width**2))
    beta_vals = beta_func(t)
    t_scaled = np.cumsum(dt ** beta_vals)

    min_len = min(len(t_scaled), len(T_internal))
    t_scaled = t_scaled[:min_len]
    T_internal = T_internal[:min_len]
    t_trimmed = t[:min_len]

    try:
        interp_beta = interp1d(t_scaled, T_internal, kind="linear", fill_value="extrapolate", bounds_error=False)
        T_beta_scaled = interp_beta(t_trimmed)
    except Exception as e:
        st.error(f"補間エラー（β補正）: {e}")
        st.stop()

    df = df.iloc[:len(T_beta_scaled)].copy()
    df["T_beta_scaled"] = T_beta_scaled

    st.sidebar.header("🧠 DTW 補正")
    if st.sidebar.button("DTW補正を実行"):
        with st.spinner("DTW実行中..."):
            try:
                T1 = pd.to_numeric(df["T_beta_scaled"], errors="coerce").astype("float64").to_numpy().ravel()
                T2 = pd.to_numeric(df["T_surface"], errors="coerce").astype("float64").to_numpy().ravel()

                if T1.ndim != 1 or T2.ndim != 1:
                    raise ValueError("DTW入力が1次元ではありません")

                mask = np.isfinite(T1) & np.isfinite(T2)
                T1_clean = T1[mask]
                T2_clean = T2[mask]

                if len(T1_clean) == 0 or len(T2_clean) == 0:
                    raise ValueError("DTW入力が空です")

                distance, path = fastdtw(T1_clean, T2_clean, dist=euclidean)
                idx_i, idx_s = zip(*path)
                t_warped = df["time"].values[np.array(idx_s)]
                T_aligned = df["T_beta_scaled"].values[np.array(idx_i)]
                interp_dtw = interp1d(t_warped, T_aligned, kind="linear", fill_value="extrapolate", bounds_error=False)
                df["T_dtw_aligned"] = interp_dtw(df["time"])

            except Exception as e:
                st.error(f"❌ DTW処理エラー: {type(e).__name__}: {e}")
                st.stop()

        st.success(f"✅ DTW完了（距離: {distance:.2f}）")

        st.sidebar.header("📐 補正式の最適化")
        if st.sidebar.button("最適化を実行"):
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
                st.info(f"a = {a_opt:.4f}, b = {b_opt:.4f}, c = {c_opt:.4f}")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df["time"], df["T_surface"], label="実測（表面）")
                ax.plot(df["time"], df["T_dtw_aligned"], label="補正（内部）", linestyle=":")
                ax.plot(df["time"], df["T_predicted"], label="推定温度", linestyle="--")
                ax.set_xlabel("時間 [s]")
                ax.set_ylabel("温度 [℃]")
                ax.legend()
                st.pyplot(fig)

                df_export = df.copy()
                for col in df_export.columns:
                    if df_export[col].dtype == "object":
                        df_export[col] = df_export[col].astype("string")

                st.download_button(
                    "📥 結果CSVダウンロード",
                    data=df_export.to_csv(index=False).encode("utf-8"),
                    file_name="temperature_result_v4.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ 最適化エラー: {type(e).__name__}: {e}")
