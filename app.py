import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="応答補正付き表面温度推定", layout="wide")
st.title("🌡️ 熱電対の応答補正＋最適推定アプリ")

st.markdown("""
内部温度（応答が遅い）を「応答補正」して、表面温度（高速応答）に近づけ、  
その上で最適な係数で表面温度を推定します。

**補正式： `T_surface ≈ a × 補正温度 + b × 補正dT/dt + c`**
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

    required_columns = {"time", "T_internal", "T_surface"}
    if not required_columns.issubset(df.columns):
        st.error(f"以下の列が必要です: {required_columns}")
        st.stop()

    df.dropna(subset=["T_internal", "T_surface"], inplace=True)

    # -----------------------------
    # 応答補正パラメータ入力
    # -----------------------------
    st.sidebar.header("📐 応答補正パラメータ")
    tau = st.sidebar.number_input("熱電対の時定数 τ [秒]", min_value=0.01, max_value=10.0, value=3.0, step=0.1)
    dt = st.sidebar.number_input("サンプリング間隔 Δt [秒]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # -----------------------------
    # 応答補正の実行： T_true ≈ T_measured + τ × dT/dt
    # -----------------------------
    df["dT_dt"] = df["T_internal"].diff() / dt
    df["dT_dt_smooth"] = df["dT_dt"].rolling(window=5, center=True).mean()
    df["T_internal_compensated"] = df["T_internal"] + tau * df["dT_dt_smooth"]
    df.dropna(inplace=True)

    # -----------------------------
    # 最適化による係数推定
    # -----------------------------
    st.sidebar.header("⚙️ 自動最適化")
    run_opt = st.sidebar.button("最適化を実行する")

    if run_opt:
        with st.spinner("最適化中..."):

            def objective(params):
                a, b, c = params
                pred = a * df["T_internal_compensated"] + b * df["dT_dt_smooth"] + c
                return ((df["T_surface"] - pred) ** 2).mean()

            res = minimize(objective, x0=[1.0, 0.0, 0.0], method='Nelder-Mead')
            a_opt, b_opt, c_opt = res.x
            df["T_surface_predicted"] = a_opt * df["T_internal_compensated"] + b_opt * df["dT_dt_smooth"] + c_opt

        st.success("✅ 最適化完了！")
        st.info(f"📌 最適係数: `a = {a_opt:.4f}`、`b = {b_opt:.4f}`、`c = {c_opt:.4f}`")

        # -----------------------------
        # グラフ描画
        # -----------------------------
        st.subheader("📊 実測 vs 補正 vs 推定")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["time"], df["T_surface"], label="実測（表面）", linewidth=2)
        ax.plot(df["time"], df["T_internal_compensated"], label="補正内部温度", linestyle=":")
        ax.plot(df["time"], df["T_surface_predicted"], label="推定（補正＋最適化）", linestyle="--")
        ax.set_xlabel("時間 [s]")
        ax.set_ylabel("温度 [℃]")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # -----------------------------
        # データ確認とダウンロード
        # -----------------------------
        st.subheader("📋 推定データ一部")
        st.dataframe(df[["time", "T_internal", "T_internal_compensated", "T_surface", "T_surface_predicted"]].head(10))

        st.download_button(
            label="📥 結果をCSVでダウンロード",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="compensated_temperature_estimation.csv",
            mime='text/csv'
        )
