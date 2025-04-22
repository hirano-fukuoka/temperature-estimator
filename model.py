import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def load_and_prepare_data(file_path, n_lags=20):
    df = pd.read_csv(file_path)
    for lag in range(1, n_lags+1):
        df[f"T_internal_lag{lag}"] = df["T_internal"].shift(lag)
    df = df.dropna()
    X = df[[f"T_internal_lag{i}" for i in range(1, n_lags+1)]]
    y = df["T_surface"]
    return X, y, df

def train_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
