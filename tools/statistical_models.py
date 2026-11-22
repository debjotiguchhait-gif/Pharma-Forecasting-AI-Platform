
import pandas as pd


def seasonal_naive_forecast(series, horizon=12, season_length=12):
    if series.isna().any():
        raise ValueError("Missing months must be resolved before modelling")

    if len(series) < 2 * season_length:
        raise ValueError("At least 24 months required")

    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")

    history = series.astype(float)
    last_season = history.iloc[-season_length:].to_numpy()

    predictions = [
        float(last_season[i % season_length])
        for i in range(horizon)
    ]

    future_index = pd.period_range(
        history.index[-1] + 1,
        periods=horizon,
        freq="M"
    )

    return pd.Series(
        predictions,
        index=future_index,
        name="forecast_items"
    )

from statsmodels.tsa.holtwinters import ExponentialSmoothing


def ets_forecast(series, horizon=12):
    if series.isna().any():
        raise ValueError("Missing months must be resolved before modelling")

    if len(series) < 36:
        raise ValueError("ETS requires at least 36 months")

    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")

    history = series.astype(float).copy()
    history.index = history.index.to_timestamp()

    model = ExponentialSmoothing(
        history,
        trend="add",
        seasonal="add",
        seasonal_periods=12,
        initialization_method="estimated"
    )

    fitted = model.fit(optimized=True)

    predictions = fitted.forecast(horizon)

    future_index = pd.period_range(
        series.index[-1] + 1,
        periods=horizon,
        freq="M"
    )

    return pd.Series(
        predictions.to_numpy(),
        index=future_index,
        name="forecast_items"
    )

from statsmodels.tsa.statespace.sarimax import SARIMAX


def sarima_forecast(series, horizon=12):
    if series.isna().any():
        raise ValueError("Missing months must be resolved before modelling")

    if len(series) < 36:
        raise ValueError("SARIMA requires at least 36 months")

    if horizon < 1:
        raise ValueError("Forecast horizon must be positive")

    history = series.astype(float).copy()
    history.index = history.index.to_timestamp()

    model = SARIMAX(
        history,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False, maxiter=200)

    predictions = fitted.forecast(steps=horizon)

    future_index = pd.period_range(
        series.index[-1] + 1,
        periods=horizon,
        freq="M"
    )

    return pd.Series(
        predictions.to_numpy(),
        index=future_index,
        name="forecast_items"
    )
