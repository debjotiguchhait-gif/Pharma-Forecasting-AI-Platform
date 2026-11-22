
from typing import Literal
from pydantic import BaseModel, Field
from orchestrator.gemini import llm


class WorkflowStep(BaseModel):
    agent: Literal["data_agent", "statistical_agent"]
    task: str


class WorkflowPlan(BaseModel):
    steps: list[WorkflowStep] = Field(
        min_length=1,
        max_length=2
    )
    product_code: str | None = None
    forecast_horizon: int = Field(default=12, ge=1, le=24)


planner_llm = llm.with_structured_output(WorkflowPlan)


def create_plan(request: str) -> dict:
    prompt = f"""
You are the workflow planner for a pharmaceutical
forecasting and decision intelligence platform.

Available operational agents:
1. data_agent: validates and profiles datasets.
2. statistical_agent: evaluates statistical models
   and generates pharmaceutical forecasts.

Rules:
- Select only agents needed to fulfil the request.
- If forecasting is requested, include data_agent first
  to validate the dataset.
- Use statistical_agent only when forecasting is requested.
- Extract the exact medicine code from the request.
- Do not invent a medicine code.
- If no code is provided, return product_code=null.
- Forecast horizon defaults to 12 months.
- Do not execute tools or fabricate results.

User request:
{request}
"""

    plan = planner_llm.invoke(prompt)
    result = plan.model_dump()

    # Enforce safe, deterministic execution order.
    agents = [step["agent"] for step in result["steps"]]

    if "statistical_agent" in agents:
        result["steps"] = [
            {
                "agent": "data_agent",
                "task": "Validate the input dataset"
            },
            {
                "agent": "statistical_agent",
                "task": "Evaluate models and forecast the product"
            }
        ]

    return result
