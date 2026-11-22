
import math

from tools.time_series import prepare_model_data
from tools.backtesting import rolling_backtest
from tools.model_registry import (
    calculate_evaluation_fingerprint,
    get_model_by_fingerprint,
    connect_registry,
)
import json
from datetime import datetime, timezone

CANDIDATE_MODELS = [
    "seasonal_naive",
    "ets",
    "sarima",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "xgboost",
]


def evaluate_models(
    file_path,
    product_code,
    horizon=12,
    min_train=36,
):
    prepared = prepare_model_data(
        file_path,
        product_code,
    )

    if prepared["status"] != "READY":
        return {
            "status": prepared["status"],
            "product_code": product_code,
            "reason": prepared.get("reason"),
            "pending_months": prepared.get("pending_months", []),
        }

    series = prepared["series"]
    fingerprint = calculate_evaluation_fingerprint(
    series=series,
    product_code=product_code,
    horizon=horizon,
    min_train=min_train,
    candidate_models=CANDIDATE_MODELS,
    approved_months=prepared.get("approved_months"),
    excluded_months=prepared.get("excluded_months"),
    )

    cached = get_model_by_fingerprint(
        product_code,
        fingerprint,
    )

    if cached is not None:
        return {
            "status": "SUCCESS",
            "product_code": product_code,
            "champion_model": cached["champion_model"],
            "challenger_model": cached["challenger_model"],
            "champion_metrics": cached["champion_metrics"],
            "challenger_metrics": cached["challenger_metrics"],
            "ranking": [
                metric
                for metric in (
                    cached["champion_metrics"],
                    cached["challenger_metrics"],
                )
                if metric is not None
            ],
            "failed_models": {},
            "excluded_months": prepared.get("excluded_months", []),
            "approved_months": prepared.get("approved_months", []),
            "evaluation_horizon": horizon,
            "evaluation_fingerprint": fingerprint,
            "registry_version": cached["version"],
            "registry_cache_hit": True,
        }
    results = []
    failures = {}

    for model_name in CANDIDATE_MODELS:
        try:
            metrics = rolling_backtest(
                series,
                model_name=model_name,
                horizon=horizon,
                min_train=min_train,
            )

            if math.isfinite(metrics["wape_pct"]):
                results.append(metrics)
            else:
                failures[model_name] = "Non-finite WAPE"

        except Exception as exc:
            failures[model_name] = str(exc)

    if not results:
        return {
            "status": "BLOCKED",
            "product_code": product_code,
            "reason": "No eligible forecasting models",
            "failures": failures,
        }

    results.sort(
        key=lambda x: (
            x["wape_pct"],
            x["mae"],
            x["rmse"],
        )
    )

    champion = results[0]
    challenger = results[1] if len(results) > 1 else None

    result = {
    "status": "SUCCESS",
    "product_code": product_code,
    "champion_model": champion["model"],
    "challenger_model": challenger["model"] if challenger else None,
    "champion_metrics": champion,
    "challenger_metrics": challenger,
    "ranking": results,
    "failed_models": failures,
    "excluded_months": prepared["excluded_months"],
    "approved_months": prepared["approved_months"],
    "evaluation_horizon": horizon,
    "evaluation_fingerprint": fingerprint,
    "registry_cache_hit": False,
    }

    conn = connect_registry()

    try:
        next_version = conn.execute(
            """
            SELECT COALESCE(MAX(version), 0) + 1
            FROM model_registry
            WHERE product_code = ?
            """,
            (str(product_code),),
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO model_registry (
                product_code,
                version,
                champion_model,
                challenger_model,
                champion_metrics,
                challenger_metrics,
                run_fingerprint,
                evaluation_fingerprint,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(product_code),
                next_version,
                champion["model"],
                challenger["model"] if challenger else None,
                json.dumps(champion),
                json.dumps(challenger) if challenger else None,
                fingerprint,
                fingerprint,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

        conn.commit()
        result["registry_version"] = next_version

    finally:
        conn.close()

    return result