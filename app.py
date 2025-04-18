import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="表面温度自動最適化アプリ", layout="wide")
st.title("🌡️ 内部温度から表面温度を自動最適化推定")

st.markdown("""
熱電対の内部温度から、時間シフト＋変化率を考慮し、  
実測表面温度と誤差が最小になるように係数 `a`, `b`, `c` を**自動で最適化**します。
""")

# -----------------------------
# ファイルアップロード
# -----------------------------
uploaded_file = st.file_uploader("📤 CSV または Excel ファイルをアップロード", type=["csv", "xlsx"])

# -----------------------------
# メイン処理
# -----------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("CSV または Excel ファイルのみ対応しています。")
            st.stop()
    except Exception as e:
        st.error(f"❌ 読み込みエラー: {e}")
        st.stop()

    st.success("✅ ファイル読み込み完了")
    st.subheader("データプレビュー")
    st.dataframe(df.head())

    # -----------------------------
    # 必須列の確認
    # -----------------------------
    required_columns = {"time", "T_internal", "T_surface"}
    if not required_columns.issubset(df.columns):
        st.error(f"以下の列が必要です: {required_columns}")
        st.stop()

    df.dropna(subset=["T_internal", "T_surface"], inplace=True)

    # -----------------------------
    # 応答補正（時間シフト）
    # -----------------------------
    st.sidebar.header("📐 応答補正設定")
    tau = st.sidebar.number_input("応答遅れ τ [秒]", min_value=0.01, max_value=10.0, value=1.5, step=0.1)
    dt = st.sidebar.number_input("サンプリング間隔 Δt [秒]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    shift_steps = int(tau / dt)
    st.sidebar.markdown(f"⏩ 時間シフト = {shift_steps} サンプル")

    # 内部温度を先送り（時間補正）
    df["T_internal_shifted"] = df["T_internal"].shift(-shift_steps)
    df["dT_dt"] = df["T_internal_shifted"].diff() / dt
    df.dropna(inplace=True)

    # -----------------------------
    # 自動最適化の実行
    # -----------------------------
    st.sidebar.header("⚙️ 自動最適化")
    run_opt = st.sidebar.button("最適化を実行する")

    if run_opt:
        with st.spinner("最適化中..."):

            def objective(params):
                a, b, c = params
                pred = a * df["T_internal_shifted"] + b * df["dT_dt"] + c
                return ((df["T_surface"] - pred) ** 2).mean()

            res = minimize(objective, x0=[1.0, 0.0, 0.0], method='Nelder-Mead')

            a_opt, b_opt, c_opt = res.x
            df["T_surface_predicted"] = a_opt * df["T_internal_shifted"] + b_opt * df["dT_dt"] + c_opt

        st.success("✅ 最適化完了！")
        st.info(f"📌 最適係数: `a = {a_opt:.4f}`、`b = {b_opt:.4f}`、`c = {c_opt:.4f}`")

        # -----------------------------
        # グラフ表示
        # -----------------------------
        st.subheader("📊 推定結果グラフ")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["time"], df["T_surface"], label="実測（表面）", linewidth=2)
        ax.plot(df["time"], df["T_surface_predicted"], label="推定（最適化）", linestyle="--")
        ax.set_xlabel("時間 [s]")
        ax.set_ylabel("温度 [℃]")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # -----------------------------
        # テーブルとCSV出力
        # -----------------------------
        st.subheader("📋 推定結果データ")
        st.dataframe(df[["time", "T_internal", "T_internal_shifted", "dT_dt", "T_surface", "T_surface_predicted"]].head(10))

        st.download_button(
            label="📥 CSVでダウンロード",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="optimized_surface_temperature.csv",
            mime='text/csv'
        )
