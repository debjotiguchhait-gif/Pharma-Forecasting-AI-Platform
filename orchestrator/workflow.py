
from typing import TypedDict
from uuid import uuid4
import json
from langgraph.graph import StateGraph, START, END
import os
from orchestrator.planner import create_plan
from agents.data_agent import validate_data
from agents.statistical_agent import run_statistical_forecast
from audit.logger import record_event
from tools.time_series import prepare_time_series

from agents.evaluation_agent import evaluate_models
from agents.ml_agent import ML_MODELS, ml_forecast
from tools.statistical_models import (
    seasonal_naive_forecast,
    ets_forecast,
    sarima_forecast,
)
from tools.time_series import prepare_training_window
from tools.time_series import prepare_model_data

from agents.monitoring_agent import monitor_forecast
from agents.business_agent import generate_business_insights

import asyncio
from orchestrator.mcp_client import get_mcp_client

class WorkflowState(TypedDict):
    request: str
    user_id: str
    run_id: str
    file_path: str
    plan: dict
    validation_result: dict
    forecast_result: dict
    status: str
    evaluation_result: dict
    monitoring_result: dict
    business_result: dict

STATISTICAL_MODELS = {
    "seasonal_naive": seasonal_naive_forecast,
    "ets": ets_forecast,
    "sarima": sarima_forecast,
}


def generate_champion_forecast(
    series,
    champion_model,
    horizon,
):
    if champion_model in ML_MODELS:
        return ml_forecast(
            series,
            horizon=horizon,
            model_name=champion_model,
        )

    if champion_model in STATISTICAL_MODELS:
        clean_series = prepare_training_window(series)

        return STATISTICAL_MODELS[champion_model](
            clean_series,
            horizon=horizon,
        )

    raise ValueError(
        f"Unknown champion model: {champion_model}"
    )

def planner_node(state: WorkflowState):
    plan = create_plan(state["request"])

    record_event(
        user_id=state["user_id"],
        action="WORKFLOW_PLANNING",
        status="SUCCESS",
        details={"run_id": state["run_id"], "plan": plan}
    )

    return {"plan": plan, "status": "PLANNED"}



def data_node(state: WorkflowState):
    plan = state["plan"]

    try:
        validation = validate_data(state["file_path"])

        # Validate the requested product independently.
        if any(
            step["agent"] == "statistical_agent"
            for step in plan["steps"]
        ):
            product_code = plan.get("product_code")

            if not product_code:
                raise ValueError("Medicine code is required")

            prepared = prepare_model_data(
                state["file_path"],
                product_code,
            )

            if prepared.get("status") not in ("READY", "SUCCESS"):
                return {
                    "validation_result": {
                        "status": prepared.get("status", "BLOCKED"),
                        "reason": prepared.get("reason"),
                        "pending_months": prepared.get("pending_months", []),
                    },
                    "status": prepared.get("status", "BLOCKED"),
                }

            series = prepared["series"]

            if series.dropna().shape[0] < 36:
                raise ValueError(
                    "Insufficient usable history: minimum 36 months"
                )

        record_event(
            user_id=state["user_id"],
            action="DATA_VALIDATION",
            status="SUCCESS",
            details={"run_id": state["run_id"]}
        )

        return {
            "validation_result": validation,
            "status": "VALIDATED"
        }

    except (ValueError, KeyError, FileNotFoundError) as exc:
        record_event(
            user_id=state["user_id"],
            action="DATA_VALIDATION",
            status="BLOCKED",
            details={
                "run_id": state["run_id"],
                "reason": str(exc)
            }
        )

        return {
            "validation_result": {
                "status": "BLOCKED",
                "reason": str(exc)
            },
            "status": "BLOCKED"
        }

def evaluation_node(state):
    plan = state["plan"]

    evaluation = evaluate_models(
        file_path=state["file_path"],
        product_code=plan["product_code"],
        horizon=12,
        min_train=36,
    )

    return {"evaluation_result": evaluation}

import json

async def run_mcp_forecast(product_code):
    client = get_mcp_client()
    tools = await client.get_tools()

    forecast_tool = next(
        tool for tool in tools
        if tool.name == "forecast_medicine"
    )

    response = await forecast_tool.ainvoke({
        "product_code": product_code
    })

    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        return json.loads(response)

    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and item.get("text"):
                return json.loads(item["text"])

    raise ValueError(f"Unsupported MCP response: {type(response)}")

async def call_mcp_tool(tool_name, arguments):
    client = get_mcp_client()
    tools = await client.get_tools()

    tool = next(
        (tool for tool in tools if tool.name == tool_name),
        None,
    )

    if tool is None:
        raise ValueError(f"MCP tool not found: {tool_name}")

    response = await tool.ainvoke(arguments)

    if isinstance(response, dict):
        return response

    if isinstance(response, str):
        return json.loads(response)

    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict) and item.get("text"):
                return json.loads(item["text"])

    raise ValueError(f"Unexpected MCP response: {type(response)}")

def forecast_node(state: WorkflowState):
    plan = state["plan"]
    product_code = plan["product_code"]
    horizon = plan["forecast_horizon"]

    evaluation = state["evaluation_result"]

    if evaluation["status"] != "SUCCESS":
        result = {
            "status": evaluation["status"],
            "reason": evaluation.get("reason"),
            "pending_months": evaluation.get("pending_months", []),
            "champion_model": None,
            "forecast": {},
        }

    else:
        prepared = prepare_model_data(
            state["file_path"],
            product_code,
        )

        forecast = generate_champion_forecast(
            series=prepared["series"],
            champion_model=evaluation["champion_model"],
            horizon=horizon,
        )

        result = {
            "status": "SUCCESS",
            "product_code": product_code,
            "champion_model": evaluation["champion_model"],
            "challenger_model": evaluation["challenger_model"],
            "evaluation": evaluation["ranking"],
            "forecast": {
                str(month): round(float(value), 2)
                for month, value in forecast.items()
            },
        }

    record_event(
        user_id=state["user_id"],
        action="STATISTICAL_FORECAST",
        status=result["status"],
        details={
            "run_id": state["run_id"],
            "product_code": plan["product_code"],
            "champion_model": result.get("champion_model")
        }
    )

    return {
        "forecast_result": result,
        "status": result["status"]
    }

async def mcp_evaluation_node(state: WorkflowState):
    print("MCP EVALUATION AGENT STARTED")
    product_code = state["plan"]["product_code"]

    evaluation = await call_mcp_tool(
        "evaluate_medicine",
        {"product_code": product_code},
    )
    print("MCP EVALUATION RESPONSE:", evaluation)

    return {
        "evaluation_result": evaluation,
        "status": evaluation["status"],
    }
    print("MCP EVALUATION RESULT:", evaluation.get("status"))

async def mcp_forecast_node(state: WorkflowState):
    print("MCP FORECAST AGENT STARTED")
    print("EVALUATION STATUS:", state.get("evaluation_result", {}).get("status"))
    evaluation = state["evaluation_result"]

    if evaluation["status"] != "SUCCESS":
        return {
            "status": evaluation["status"],
            "forecast_result": {
                "status": evaluation["status"],
                "reason": evaluation.get("reason"),
            },
        }

    product_code = state["plan"]["product_code"]

    result = await call_mcp_tool(
        "forecast_medicine",
        {
            "product_code": product_code,
            "champion_model": evaluation["champion_model"],
            "challenger_model": evaluation["challenger_model"],
        },
    )
    print("MCP FORECAST RESPONSE:", result)
    return {
        "forecast_result": result,
        "status": result["status"],
    }

def audit_node(state: WorkflowState):
    forecast = state.get("forecast_result") or {}
    validation = state.get("validation_result") or {}
    evaluation = state.get("evaluation_result") or {}

    reason = (
        forecast.get("reason")
        or validation.get("reason")
        or evaluation.get("reason")
    )

    pending_months = (
        validation.get("pending_months")
        or evaluation.get("pending_months")
        or []
    )

    record_event(
        user_id=state["user_id"],
        action="FORECAST_AUDIT",
        status=state["status"],
        details={
            "run_id": state["run_id"],
            "product_code": state["plan"].get("product_code"),
            "champion_model": forecast.get("champion_model"),
            "challenger_model": forecast.get("challenger_model"),
            "reason": reason,
            "pending_months": pending_months,
            "forecast_generated": forecast.get("status") == "SUCCESS",
        },
    )

    return {}

def monitoring_node(state: WorkflowState):
    result = state.get("forecast_result", {})

    if result.get("status") != "SUCCESS":
        return {"monitoring_result": {"status": "NOT_APPLICABLE"}}

    evaluation = state.get("evaluation_result", {})
    metrics = evaluation.get("champion_metrics", {})

    prepared = prepare_model_data(
        state["file_path"],
        state["plan"]["product_code"],
    )

    latest_month = str(prepared["series"].index.max())

    monitoring = monitor_forecast(
        latest_actual_month=latest_month,
        last_training_month=latest_month,
        current_wape=metrics.get("wape_pct"),
    )

    return {"monitoring_result": monitoring}

def business_node(state: WorkflowState):
    forecast_result = state.get("forecast_result", {})

    if forecast_result.get("status") != "SUCCESS":
        return {
            "business_result": {"status": "NOT_APPLICABLE"}
        }

    insights = generate_business_insights(
        forecast_result["forecast"]
    )

    return {"business_result": insights}

def after_planner(state: WorkflowState):
    agents = [
        step["agent"]
        for step in state["plan"]["steps"]
    ]

    return "data_agent" if "data_agent" in agents else "end"



def after_data(state):
    status = state.get("status")

    if status in ("READY", "VALIDATED"):
        return "mcp_evaluation_agent"

    return "audit_agent"


builder = StateGraph(WorkflowState)

builder.add_node("planner", planner_node)
builder.add_node("data_agent", data_node)
builder.add_node("evaluation_agent", evaluation_node)
builder.add_node("statistical_agent", forecast_node)
builder.add_node("audit_agent", audit_node)
builder.add_node("monitoring_agent", monitoring_node)
builder.add_node("business_agent", business_node)
builder.add_node("mcp_forecast_agent", mcp_forecast_node)
builder.add_node("mcp_evaluation_agent", mcp_evaluation_node)

builder.add_edge(START, "planner")

builder.add_conditional_edges(
    "planner",
    after_planner,
    {
        "data_agent": "data_agent",
        "end": END
    }
)

builder.add_conditional_edges(
    "data_agent",
    after_data,
    {
        "mcp_evaluation_agent": "mcp_evaluation_agent",
        "audit_agent": "audit_agent",
    },
)

def after_mcp_evaluation(state):
    print("EVALUATION ROUTING STATUS:", state.get("status"))

    if state.get("status") == "SUCCESS":
        return "forecast"

    return "audit"

builder.add_conditional_edges(
    "mcp_evaluation_agent",
    after_mcp_evaluation,
    {
        "forecast": "mcp_forecast_agent",
        "audit": "audit_agent",
    },
)
builder.add_edge("mcp_forecast_agent", "monitoring_agent")
builder.add_edge("evaluation_agent", "statistical_agent")
builder.add_edge("statistical_agent", "monitoring_agent")
builder.add_edge("monitoring_agent", "business_agent")
builder.add_edge("business_agent", "audit_agent")
builder.add_edge("audit_agent", END)

workflow = builder.compile()



async def run_workflow(request, user_id="local_user"):
    initial_state = {
        "request": request,
        "user_id": user_id,
        "run_id": str(uuid4()),
        "file_path": os.environ.get(
            "PHARMA_DATA_PATH",
            "data/raw/nhs_monthly_prescribing.csv",
        ),
        "plan": {},
        "validation_result": {},
        "forecast_result": {},
        "status": "PENDING",
    }

    return await workflow.ainvoke(initial_state)
