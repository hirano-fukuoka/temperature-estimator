import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def create_lag_features(df, n_lags=20):
    for lag in range(1, n_lags+1):
        df[f"T_internal_lag{lag}"] = df["T_internal"].shift(lag)
    return df.dropna()

def prepare_train_data(df, n_lags=20):
    df_lag = create_lag_features(df.copy(), n_lags)
    X = df_lag[[f"T_internal_lag{i}" for i in range(1, n_lags+1)]]
    y = df_lag["T_surface"]
    return X, y

def prepare_predict_data(df, n_lags=20):
    df_lag = create_lag_features(df.copy(), n_lags)
    X = df_lag[[f"T_internal_lag{i}" for i in range(1, n_lags+1)]]
    return X, df_lag
