
from orchestrator.workflow import run_workflow
import hashlib
import inspect
import json
from pathlib import Path
from datetime import datetime, timezone
import asyncio

from agents.evaluation_agent import evaluate_models
from tools.time_series import prepare_model_data
from orchestrator.workflow import generate_champion_forecast


DATA_PATH = "data/raw/nhs_monthly_prescribing.csv"
OUTPUT_PATH = Path("outputs/batch_forecasts.jsonl")

def get_run_fingerprint():
    from agents import evaluation_agent, ml_agent
    from tools import (
        backtesting,
        statistical_models,
        time_series,
        hitl_review,
    )

    config = {
        "horizon": 12,
        "min_train": 36,
        "model_set": [
            "seasonal_naive", "ets", "sarima",
            "ridge", "random_forest",
            "gradient_boosting", "xgboost",
        ],
    }

    digest = hashlib.sha256()

    for path in [
        DATA_PATH,
        "outputs/anomaly_reviews.jsonl",
        "mcp_server.py",
        "orchestrator/workflow.py",
        "orchestrator/mcp_client.py",
        "main.py",
    ]:
        file = Path(path)
        digest.update(path.encode())

        if file.exists():
            digest.update(file.read_bytes())
        else:
            digest.update(b"MISSING")

    digest.update(
        json.dumps(config, sort_keys=True).encode()
    )

    for module in [
        evaluation_agent,
        ml_agent,
        backtesting,
        statistical_models,
        time_series,
        hitl_review,
    ]:
        digest.update(inspect.getsource(module).encode())

    return digest.hexdigest()

def run_batch(product_codes):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    run_fingerprint = get_run_fingerprint()
    completed = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    if (
                        record.get("status") == "SUCCESS"
                        and record.get("run_fingerprint") == run_fingerprint
                    ):
                        completed.add(record["product_code"])

    for product_code in product_codes:
        product_code = str(product_code)

        if product_code in completed:
            print(f"SKIPPED: {product_code} (already completed)")
            continue

        print(f"PROCESSING: {product_code}", flush=True)

        record = {
            "product_code": product_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": run_fingerprint,
        }

        try:
            evaluation = evaluate_models(
                DATA_PATH,
                product_code,
                horizon=12,
                min_train=36,
            )

            record["status"] = evaluation["status"]

            if evaluation["status"] == "SUCCESS":
                prepared = prepare_model_data(
                    DATA_PATH,
                    product_code,
                )

                forecast = generate_champion_forecast(
                    prepared["series"],
                    evaluation["champion_model"],
                    horizon=12,
                )

                record.update({
                    "champion_model": evaluation["champion_model"],
                    "challenger_model": evaluation["challenger_model"],
                    "champion_metrics": evaluation["champion_metrics"],
                    "challenger_metrics": evaluation["challenger_metrics"],
                    "forecast": {
                        str(month): round(float(value), 2)
                        for month, value in forecast.items()
                    },
                    "excluded_months": evaluation["excluded_months"],
                })

            else:
                record["reason"] = evaluation.get("reason")
                record["pending_months"] = evaluation.get(
                    "pending_months", []
                )

        except Exception as exc:
            record["status"] = "FAILED"
            record["reason"] = str(exc)

        with OUTPUT_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print(
            product_code,
            record["status"],
            record.get("champion_model"),
            flush=True,
        )
        if record["status"] == "SUCCESS":
            completed.add(product_code)

def main():
    request = (
        "Validate the dataset and forecast medicine "
        "0101021B0 for the next 12 months"
    )

    result = asyncio.run(run_workflow(request))

    print("\nBUSINESS INSIGHTS")
    print(result.get("business_result", {}))

    print("\nMONITORING")
    print(result.get("monitoring_result", {}))

    print("\nPHARMA FORECASTING PLATFORM")
    print("Run ID:", result["run_id"])
    print("Status:", result["status"])

    print("\nEXECUTION PLAN")
    for step in result["plan"]["steps"]:
        print("-", step["agent"])

    forecast = result.get("forecast_result", {})

    if forecast:
        print("\nFORECAST RESULT")
        print("Status:", forecast.get("status"))
        print("Champion:", forecast.get("champion_model"))
        print("Challenger:", forecast.get("challenger_model"))

        for month, value in forecast.get(
            "forecast", {}
        ).items():
            print(month, value)

import pandas as pd


def discover_eligible_products(
    file_path=DATA_PATH,
    min_months=36,
):
    df = pd.read_csv(
        file_path,
        dtype={"bnf_chemical_substance_code": str},
    )

    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M")

    history = df.groupby("bnf_chemical_substance_code")["month"].nunique()

    eligible = history[history >= min_months].index.tolist()

    print(f"Total products: {len(history)}")
    print(f"Eligible products: {len(eligible)}")

    return eligible
async def test_mcp_batch():
    products = [
        "0212000B0",  # Previously successful
        "0101021B0",  # HITL review required
        "0102000A0",  # Previously successful
    ]

    results = []

    for product_code in products:
        print(f"\nPROCESSING: {product_code}")

        request = (
            f"Forecast medicine {product_code} "
            "for the next 12 months"
        )

        try:
            result = await run_workflow(request)

            forecast = result.get("forecast_result") or {}
            status = result.get("status")

            print("STATUS:", status)
            print("CHAMPION:", forecast.get("champion_model"))

            results.append({
                "product_code": product_code,
                "run_id": result.get("run_id"),
                "status": status,
                "champion_model": forecast.get("champion_model"),
            })

        except Exception as exc:
            print("ERROR:", str(exc))
            results.append({
                "product_code": product_code,
                "status": "ERROR",
                "reason": str(exc),
            })

    print("\nBATCH SUMMARY")
    for item in results:
        print(item)

async def run_mcp_batch(product_codes):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fingerprint = get_run_fingerprint()
    completed = set()

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue

                record = json.loads(line)

                if (
                    record.get("status") == "SUCCESS"
                    and record.get("run_fingerprint") == fingerprint
                    and record.get("execution_mode") == "MCP_LANGGRAPH"
                ):
                    completed.add(record["product_code"])

    for product_code in product_codes:
        product_code = str(product_code)

        if product_code in completed:
            print(f"SKIPPED: {product_code} (checkpoint)")
            continue

        print(f"\nPROCESSING: {product_code}", flush=True)

        record = {
            "product_code": product_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_fingerprint": fingerprint,
            "execution_mode": "MCP_LANGGRAPH",
        }

        try:
            request = (
                f"Forecast medicine {product_code} "
                "for the next 12 months"
            )

            result = await run_workflow(request)

            forecast = result.get("forecast_result") or {}
            evaluation = result.get("evaluation_result") or {}
            validation = result.get("validation_result") or {}

            record.update({
                "run_id": result.get("run_id"),
                "status": result.get("status"),
                "champion_model": forecast.get("champion_model"),
                "challenger_model": forecast.get("challenger_model"),
                "forecast": forecast.get("forecast", {}),
                "champion_metrics": evaluation.get("champion_metrics"),
                "challenger_metrics": evaluation.get("challenger_metrics"),
                "excluded_months": evaluation.get("excluded_months", []),
                "reason": (
                    forecast.get("reason")
                    or validation.get("reason")
                    or evaluation.get("reason")
                ),
                "pending_months": (
                    validation.get("pending_months")
                    or evaluation.get("pending_months")
                    or []
                ),
            })

        except Exception as exc:
            record["status"] = "FAILED"
            record["reason"] = str(exc)

        with OUTPUT_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

        print(
            product_code,
            record["status"],
            record.get("champion_model"),
            flush=True,
        )

        if record["status"] == "SUCCESS":
            completed.add(product_code)

if __name__ == "__main__":
    with open(
        "outputs/imputation_validation.json",
        encoding="utf-8"
    ) as file:
        validation_results = json.load(file)

    products = [
        record["product_code"]
        for record in validation_results
        if record["status"] == "VALIDATED"
    ]

    print("Isolated-gap products:", len(products))

    asyncio.run(run_mcp_batch(products))