
import asyncio
import json
from pathlib import Path

from audit.logger import record_event
from orchestrator.workflow import run_workflow
from tools.monthly_ingestion import ingest_monthly_data


async def reassess_monthly_data(incoming_path, commit=False):
    ingestion = ingest_monthly_data(
        incoming_path,
        commit=commit,
    )
    if ingestion["status"] == "ALREADY_INGESTED":
        return {
            "status": "SKIPPED",
            "reason": "Monthly data already ingested",
            "ingestion": ingestion,
            "reassessment": [],
        }

    if not commit:
        return {
            "status": "PREVIEW",
            "ingestion": ingestion,
            "reassessment": [],
        }

    results = []

    for product_code in ingestion["affected_products"]:
        try:
            result = await run_workflow(
                f"Forecast medicine {product_code} for the next 12 months",
                user_id="monthly_ingestion_system",
            )

            evaluation = result.get("evaluation_result") or {}
            forecast = result.get("forecast_result") or {}

            outcome = {
                "product_code": product_code,
                "status": result.get("status"),
                "champion_model": evaluation.get("champion_model"),
                "registry_version": evaluation.get("registry_version"),
                "registry_cache_hit": evaluation.get("registry_cache_hit"),
                "forecast_generated": (
                    forecast.get("status") == "SUCCESS"
                ),
            }

        except Exception as exc:
            outcome = {
                "product_code": product_code,
                "status": "FAILED",
                "reason": str(exc),
                "forecast_generated": False,
            }

        results.append(outcome)

    summary = {
        "status": "COMPLETED",
        "transaction_id": ingestion["transaction_id"],
        "affected_product_count": len(results),
        "results": results,
    }

    record_event(
        user_id="monthly_ingestion_system",
        action="MONTHLY_MODEL_REASSESSMENT",
        status="COMPLETED",
        details=summary,
    )

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python -m tools.monthly_reassessment incoming.csv"
        )

    result = asyncio.run(
        reassess_monthly_data(
            incoming_path=sys.argv[1],
            commit=False,
        )
    )

    print(json.dumps(result, indent=2))
