
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from tools.time_series import prepare_training_window

ML_MODELS = [
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]

LAGS = [1, 2, 3, 6, 12]

FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + ["rolling_3", "rolling_6", "month"]
)


def build_features(series):
    df = pd.DataFrame({
        "target": series.astype(float)
    })

    for lag in LAGS:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    previous = df["target"].shift(1)

    df["rolling_3"] = previous.rolling(3).mean()
    df["rolling_6"] = previous.rolling(6).mean()
    df["month"] = df.index.month

    return df.dropna()


def get_model(model_name):
    models = {
        "ridge": lambda: make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        ),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=150,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            random_state=42,
        ),
        "xgboost": lambda: XGBRegressor(
            n_estimators=150,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=2,
        ),
    }

    if model_name not in models:
        raise ValueError(f"Unknown ML model: {model_name}")

    return models[model_name]()


def ml_forecast(series, horizon=12, model_name="ridge"):
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")

    clean = prepare_training_window(series)

    if not isinstance(clean.index, pd.PeriodIndex):
        raise ValueError("Expected monthly PeriodIndex")

    training = build_features(clean)

    if len(training) < 24:
        raise ValueError("Insufficient ML training history")

    model = get_model(model_name)

    model.fit(
        training[FEATURES],
        training["target"],
    )

    history = clean.copy()
    predictions = {}

    for _ in range(horizon):
        next_month = history.index[-1] + 1

        features = {
            f"lag_{lag}": float(history.iloc[-lag])
            for lag in LAGS
        }

        features["rolling_3"] = float(
            history.iloc[-3:].mean()
        )
        features["rolling_6"] = float(
            history.iloc[-6:].mean()
        )
        features["month"] = next_month.month

        X_next = pd.DataFrame(
            [features],
            columns=FEATURES,
        )

        prediction = max(
            0.0,
            float(model.predict(X_next)[0]),
        )

        predictions[next_month] = prediction
        history.loc[next_month] = prediction

    return pd.Series(predictions, dtype=float)
