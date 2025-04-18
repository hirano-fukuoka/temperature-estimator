import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from io import BytesIO

# -----------------------------
# ページ設定
# -----------------------------
st.set_page_config(page_title="金型表面温度推定アプリ", layout="wide")
st.title("🌡️ 金型内部温度から表面温度を推定するアプリ")

st.markdown("""
このアプリでは、応答遅れのある熱電対（内部温度）を補正し、  
高速な赤外線センサーの表面温度を推定します。  
**CSV または Excel ファイル**をアップロードしてご利用ください。
""")

# -----------------------------
# ファイルアップロード
# -----------------------------
uploaded_file = st.file_uploader("📤 ファイルをアップロード", type=["csv", "xlsx"])

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
# ファイル読み込み＆処理
# -----------------------------
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        else:
            st.error("対応していないファイル形式です。CSVまたはExcelを使用してください。")
            st.stop()
    except Exception as e:
        st.error(f"ファイルの読み込みに失敗しました: {e}")
        st.stop()

    st.success("✅ ファイルを読み込みました")
    st.subheader("データプレビュー")
    st.dataframe(df.head())

    # -----------------------------
    # 入力チェック
    # -----------------------------
    required_columns = {"time", "T_internal", "T_surface"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ ファイルに以下の列が含まれている必要があります: {required_columns}")
        st.stop()

    # -----------------------------
    # パラメータ設定
    # -----------------------------
    st.sidebar.header("📐 応答補正パラメータ")
    tau = st.sidebar.slider("熱電対の応答遅れ τ [秒]", 1.0, 10.0, 5.0)
