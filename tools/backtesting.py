import numpy as np

from tools.statistical_models import (
    seasonal_naive_forecast,
    ets_forecast,
    sarima_forecast,
)
from tools.time_series import prepare_training_window
from agents.ml_agent import ML_MODELS, ml_forecast


STATISTICAL_MODELS = {
    "seasonal_naive": seasonal_naive_forecast,
    "ets": ets_forecast,
    "sarima": sarima_forecast,
}


def forecast_with_model(series, model_name, horizon):
    """Use a common interface for statistical and ML candidates."""
    if model_name in STATISTICAL_MODELS:
        return STATISTICAL_MODELS[model_name](series, horizon=horizon)
    if model_name in ML_MODELS:
        return ml_forecast(series, horizon=horizon, model_name=model_name)
    raise ValueError(f"Unknown model: {model_name}")


def rolling_backtest(series, model_name="seasonal_naive", horizon=3, min_train=36):
    if model_name not in STATISTICAL_MODELS and model_name not in ML_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if horizon < 1 or min_train < 1:
        raise ValueError("horizon and min_train must be positive")
    if len(series) < min_train + horizon:
        raise ValueError("Insufficient history")

    actuals = []
    predictions = []
    scored_windows = 0

    for train_end in range(min_train, len(series) - horizon + 1):
        train = series.iloc[:train_end]
        test = series.iloc[train_end:train_end + horizon]

        # Imputation is fitted to the training window, never to future targets.
        clean_train = prepare_training_window(train)
        forecast = forecast_with_model(clean_train, model_name, horizon)

        valid = test.notna().to_numpy()
        if not valid.any():
            continue

        actuals.extend(test.to_numpy(dtype=float)[valid])
        predictions.extend(forecast.to_numpy(dtype=float)[valid])
        scored_windows += 1

    if not actuals:
        raise ValueError("No valid observations for evaluation")

    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    errors = actuals - predictions

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    denominator = float(np.sum(np.abs(actuals)))
    wape = float(np.sum(np.abs(errors)) / denominator * 100) if denominator > 0 else float("nan")

    return {
        "model": model_name,
        "horizon": horizon,
        "backtest_windows": len(series) - min_train - horizon + 1,
        "scored_windows": scored_windows,
        "scored_observations": len(actuals),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "wape_pct": round(wape, 2),
    }