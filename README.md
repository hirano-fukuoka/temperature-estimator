# temperature-estimator
# 金型表面温度推定アプリ

内部センサ（熱電対）の応答補正を行い、表面温度（光ファイバー）を推定するStreamlitアプリです。

## 使用方法

1. CSVまたはExcelで以下の列を含むファイルを用意：
   - `time`: 時間（秒）
   - `T_internal`: 熱電対の温度データ
   - `T_surface`: 表面温度データ

2. [Streamlitアプリ（後でデプロイ）](https://share.streamlit.io/...) にアクセスし、ファイルをアップロード。
