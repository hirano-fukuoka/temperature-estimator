import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="金型表面温度推定アプリ", layout="wide")
st.title("🌡️ 金型内部温度から表面温度を推定するアプリ")

st.markdown("""
このアプリでは、応答遅れのある熱電対のデータを補正し、  
表面温度（赤外線センサー想定）を推定します。  
補正後の内部温度から、**自動回帰**または**手動係数指定**で推定可能です。
""")

# -----------------------------
# ファイルアップロード
# -----------------------------
uploaded_file = st.file_uploader("📤 ファイルをアップロード（CSVまたはExcel）", type=["csv", "xlsx"])

# -----------------------------
# 応答補正関数（1次遅れ逆モデル）
# -----------------------------
def correct_response(measured, alpha):
    estimated = [measured.iloc[0]]
    for t in range(1, len(measured)):
        try:
            T_est = (measured.iloc[t] - (1 - alpha) * measured.iloc[t - 1]) / alpha
        except ZeroDivisionError:
            T_est = measured.iloc[t]
        estimated.append(T_est)
    return estimated

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
        st.error(f"❌ ファイルの読み込み中にエラーが発生しました: {e}")
        st.stop()

    st.success("✅ ファイルを読み込みました")
    st.subheader("データプレビュー")
    st.dataframe(df.head())

    # -----------------------------
    # 必須列の確認
    # -----------------------------
    required_columns = {"time", "T_internal", "T_surface"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ 以下の列が必要です: {required_columns}")
        st.stop()

    # 欠損値除去
    df.dropna(subset=["T_internal", "T_surface"], inplace=True)

    # -----------------------------
    # 応答遅れ補正パラメータ
    # -----------------------------
    st.sidebar.header("📐 応答補正設定")
    tau = st.sidebar.number_input("熱電対の応答遅れ τ [秒]", min_value=0.01, max_value=10.0, value=5.0, step=0.1)
    dt = st.sidebar.number_input("サンプリング間隔 Δt [秒]", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    alpha = dt / (tau + dt)
    st.sidebar.markdown(f"補正係数 α = `{alpha:.4f}`")

    # -----------------------------
    # 応答補正処理
    # -----------------------------
    df["T_internal_corrected"] = correct_response(df["T_internal"], alpha)

    # -----------------------------
    # 表面温度推定方法選択
    # -----------------------------
    st.sidebar.header("🛠 表面温度推定モード")
    manual_mode = st.sidebar.checkbox("手動で補正係数を指定する", value=False)

    if manual_mode:
        a_coeff = st.sidebar.number_input("傾き a", value=1.0, step=0.1, format="%.2f")
        b_offset = st.sidebar.number_input("オフセット b", value=0.0, step=0.1, format="%.2f")
        df["T_surface_predicted"] = a_coeff * df["T_internal_corrected"] + b_offset
        st.info(f"📌 補正式: `T_surface_estimated = {a_coeff} × T_internal_corrected + {b_offset}`")
    else:
        model = LinearRegression()
        model.fit(df[["T_internal_corrected"]], df["T_surface"])
        df["T_surface_predicted"] = model.predict(df[["T_internal_corrected"]])
        st.success("✅ 自動回帰モデルで表面温度を推定しました")

    # -----------------------------
    # グラフ表示
    # -----------------------------
    st.subheader("📊 推定結果グラフ")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["time"], df["T_surface"], label="実測（表面）", linewidth=2)
    ax.plot(df["time"], df["T_surface_predicted"], label="推定（補正後）", linestyle="--")
    ax.set_xlabel("時間 [s]")
    ax.set_ylabel("温度 [℃]")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # -----------------------------
    # テーブル表示とCSV出力
    # -----------------------------
    st.subheader("📋 推定データの一部")
    st.dataframe(df[["time", "T_internal", "T_internal_corrected", "T_surface", "T_surface_predicted"]].head(10))

    st.download_button(
        label="📥 推定結果をCSVでダウンロード",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="estimated_surface_temperature.csv",
        mime='text/csv'
    )
