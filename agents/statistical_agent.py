
from tools.time_series import prepare_time_series
from tools.statistical_models import (
    seasonal_naive_forecast,
    ets_forecast,
    sarima_forecast
)
from tools.backtesting import rolling_backtest
from tools.anomaly_detection import detect_yoy_anomalies
from tools.hitl_review import (
    get_latest_review,
    observation_hash
)
from tools.time_series import prepare_training_window


MODELS = {
    "seasonal_naive": seasonal_naive_forecast,
    "ets": ets_forecast,
    "sarima": sarima_forecast
}



from tools.time_series import prepare_model_data
from tools.backtesting import rolling_backtest


def run_statistical_forecast(
    file_path,
    product_code,
    horizon=12,
    evaluation_horizon=3,
    min_train=36,
    training_end=None
):
    prepared = prepare_model_data(
        file_path,
        product_code
    )

    if prepared["status"] != "READY":
        return {
            "status": prepared["status"],
            "product_code": product_code,
            "reason": prepared.get("reason"),
            "pending_months": prepared.get("pending_months", []),
            "champion_model": None,
            "forecast": {}
        }

    series = prepared["series"]

    if training_end:
        series = series.loc[:training_end]

    # Excluded observations remain missing.

    if len(series) < min_train + evaluation_horizon:
        return {
            "status": "BLOCKED",
            "reason": "Insufficient history"
        }

    evaluations = []

    for name in MODELS:
        metrics = rolling_backtest(
            series,
            model_name=name,
            horizon=evaluation_horizon,
            min_train=min_train
        )
        evaluations.append(metrics)

    champion = min(
        evaluations,
        key=lambda result: result["wape_pct"]
    )

    clean_series = prepare_training_window(series)

    forecast = MODELS[champion["model"]](
        clean_series,
        horizon=horizon
    )

    return {
        "status": "SUCCESS",
        "product_code": product_code,
        "champion_model": champion["model"],
        "evaluation": evaluations,
        "approved_months": prepared["approved_months"],
        "excluded_months": prepared["excluded_months"],
        "forecast": {
            str(month): round(float(value), 2)
            for month, value in forecast.items()
        }
    }



    # Check the full observed history before any training cutoff.


    anomaly_check = detect_yoy_anomalies(
        series,
        threshold_pct=50
    )

    pending = []
    rejected_months = []

    for anomaly in anomaly_check["anomalies"]:
        month = anomaly["month"]
        review = get_latest_review(product_code, month)

        current_hash = observation_hash(
            product_code,
            month,
            anomaly["actual_items"]
        )

        if (
            review is None
            or review.get("observation_hash") != current_hash
        ):
            pending.append(month)

        elif review["decision"] == "REJECTED":
            rejected_months.append(month)

    if pending:
        return {
            "status": "REVIEW_REQUIRED",
            "product_code": product_code,
            "pending_months": pending,
            "rejected_months": rejected_months,
            "champion_model": None,
            "forecast": {}
        }

    # Exclude rejected observations from modelling,
    # but preserve their calendar positions.
    series = series.copy()

    for month in rejected_months:
        series.loc[month] = float("nan")


    if training_end:
        series = series.loc[:training_end]

    if len(series) < min_train + evaluation_horizon:
        return {
            "status": "BLOCKED",
            "reason": "Insufficient historical observations"
        }

    if series.isna().any():
        return {
            "status": "BLOCKED",
            "reason": "Missing months require review"
        }

    evaluations = []

    for name in MODELS:
        metrics = rolling_backtest(
            series,
            model_name=name,
            horizon=evaluation_horizon,
            min_train=min_train
        )
        evaluations.append(metrics)

    champion = min(
        evaluations,
        key=lambda result: result["wape_pct"]
    )

    clean_series = prepare_training_window(series)

    forecast = MODELS[champion["model"]](
        clean_series,
        horizon=horizon
    )

    return {
        "status": "SUCCESS",
        "product_code": product_code,
        "training_end": str(series.index[-1]),
        "champion_model": champion["model"],
        "evaluation": evaluations,
        "forecast": {
            str(month): round(float(value), 2)
            for month, value in forecast.items()
        }
    }
