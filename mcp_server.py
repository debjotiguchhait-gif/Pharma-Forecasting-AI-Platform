from mcp.server.fastmcp import FastMCP
import os
from agents.evaluation_agent import evaluate_models
from tools.time_series import (
    prepare_model_data,
    prepare_training_window,
)
from orchestrator.workflow import generate_champion_forecast

mcp = FastMCP("Pharma Forecasting")

DATA_PATH = os.environ.get(
    "PHARMA_DATA_PATH",
    "data/raw/nhs_monthly_prescribing.csv",
)

@mcp.tool()
def evaluate_medicine(product_code: str) -> dict:
    """Evaluate forecasting models for a medicine."""
    return evaluate_models(
        file_path=DATA_PATH,
        product_code=product_code,
        horizon=12,
        min_train=36,
    )


@mcp.tool()
def forecast_medicine(
    product_code: str,
    champion_model: str,
    challenger_model: str,
) -> dict:
    """Generate a forecast using previously evaluated models."""

    prepared = prepare_model_data(DATA_PATH, product_code)

    if prepared.get("status") not in ("READY", "SUCCESS"):
        return {
            "status": prepared["status"],
            "reason": prepared.get("reason"),
        }

    clean_series = prepare_training_window(
    prepared["series"]
    )

    forecast = generate_champion_forecast(
        series=clean_series,
        champion_model=champion_model,
        horizon=12,
    )

    return {
        "status": "SUCCESS",
        "product_code": product_code,
        "champion_model": champion_model,
        "challenger_model": challenger_model,
        "forecast": {
            str(month): round(float(value), 2)
            for month, value in forecast.items()
        },
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")