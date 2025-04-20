import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from scipy.optimize import minimize

st.set_page_config(page_title="時間スパン補正アプリ", layout="wide")
st.title("⏱ 時間スパン補正 & DTWによる表面温度推定")

st.markdown("""
このアプリは、**内部温度（遅応答）→ 表面温度（速応答）**への変換を次の方法で行います：

- `β(t)` による**時間スケーリングの局所変形**
- **Dynamic Time Warping** による時系列整列
- 温度スケーリング（a, b, c）による推定補正
""")

# --- ファイル読み込み
uploaded_file = st.file_uploader("📤 CSVまたはExcelファイルをアップロード", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file, engine="openpyxl")

    st.success("✅ ファイル読み込み完了")
    st.dataframe(df.head())

    # --- 必須列チェック
    required = {"time", "T_internal", "T_surface"}
    if not required.issubset(df.columns):
        st.error(f"必要な列が含まれていません: {required}")
        st.stop()

    df.dropna(inplace=True)
    t = df["time"].values
    T_internal = df["T_internal"].values
    T_surface = df["T_surface"].values

    dt = np.mean(np.diff(t))

    # --- β(t) 時間スパン補正
    st.sidebar.header("⏳ β(t) スパン補正設定")
    peak_center = st.sidebar.slider("ピーク中心時間 [s]", float(t[0]), float(t[-1]), float(t[len(t)//2]), step=0.1)
    peak_width = st.sidebar.slider("ピーク幅 [秒]", 0.1, 20.0, 5.0, step=0.1)
    beta_base = st.sidebar.slider("ベースβ", 1.0, 2.0, 1.0, step=0.1)
    beta_peak = st.sidebar.slider("ピーク付近β", 0.1, 1.0, 0.6, step=0.05)

    def beta_t(t):
        return beta_peak + (beta_base - beta_peak) * np.exp(-((t - peak_center) ** 2) / (2 * peak_width ** 2))

    beta_vals = beta_t(t)
    t_transformed = np.cumsum(dt ** beta_vals)  # 時間拡縮の累積（簡易変換）

    # 補間して元の時間に再サンプリング
    interp_func = interp1d(t_transformed, T_internal[:len(t_transformed)], fill_value="extrapolate")
    T_beta_scaled = interp_func(t[:len(t_transformed)])

    df = df.iloc[:len(T_beta_scaled)].copy()
    df["T_beta_scaled"] = T_beta_scaled

    # --- DTWで整列
    st.sidebar.header("🧠 Dynamic Time Warping")
    run_dtw = st.sidebar.button("DTW補正を実行")

    if run_dtw:
        with st.spinner("DTW処理中..."):
            distance, path = fastdtw(df["T_beta_scaled"].values, df["T_surface"].values, dist=euclidean)
            idx_internal, idx_surface = zip(*path)

            t_surface_warped = df["time"].values[np.array(idx_surface)]
            T_internal_warped = df["T_beta_scaled"].values[np.array(idx_internal)]

            # 補間して元の時間に合わせる
            dtw_interp = interp1d(t_surface_warped, T_internal_warped, kind="linear", fill_value="extrapolate")
            df["T_internal_dtw"] = dtw_interp(df["time"])

        st.success(f"DTW補正完了（距離: {distance:.4f}）")

        # --- a, b, cの最適化
        st.sidebar.header("📐 推定補正の最適化")
        optimize_model = st.sidebar.button("最適化を実行")

        if optimize_model:
            with st.spinner("最適化中..."):

                def objective(params):
                    a, b, c = params
                    pred = a * df["T_internal_dtw"] + b * np.gradient(df["T_internal_dtw"], dt) + c
                    return np.mean((df["T_surface"] - pred) ** 2)

                res = minimize(objective, x0=[1.0, 0.0, 0.0], method="Nelder-Mead")
                a_opt, b_opt, c_opt = res.x
                df["T_surface_predicted"] = a_opt * df["T_internal_dtw"] + b_opt * np.gradient(df["T_internal_dtw"], dt) + c_opt

            st.success("最適化完了")
            st.info(f"📌 a = {a_opt:.4f}, b = {b_opt:.4f}, c = {c_opt:.4f}")

            # --- グラフ表示
            st.subheader("📈 温度比較")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["time"], df["T_surface"], label="実測（表面）")
            ax.plot(df["time"], df["T_internal_dtw"], label="内部温度（DTW補正）", linestyle=":")
            ax.plot(df["time"], df["T_surface_predicted"], label="推定温度", linestyle="--")
            ax.set_xlabel("時間 [s]")
            ax.set_ylabel("温度 [℃]")
            ax.legend()
            st.pyplot(fig)

            # --- CSV出力
            st.download_button(
                label="📥 結果をCSVでダウンロード",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="dtw_scaled_surface_prediction.csv",
                mime="text/csv"
            )
