
import json
import os
import re

from google import genai
from google.genai import types


MODEL = os.getenv("PHARMA_GEMINI_MODEL", "gemini-3.6-flash")


from datetime import datetime

def check_unavailable_forecast(question, forecasts):
    months = {
        "january": 1, "february": 2, "march": 3,
        "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9,
        "october": 10, "november": 11, "december": 12
    }

    match = re.search(
        r"\b(" + "|".join(months) + r")\s+(\d{4})\b",
        question.lower()
    )

    if not match:
        return None

    month = months[match.group(1)]
    year = int(match.group(2))
    period = f"{year}-{month:02d}"

    if period not in forecasts:
        return (
            f"No stored forecast is available for "
            f"{match.group(1).title()} {year}."
        )

    return None

def answer_question(question, product_code, forecast_data, model_data):
    """Answer using stored records only; never generate numerical results."""

    forecast_result = (forecast_data or {}).get("forecast_result", {})
    forecasts = forecast_result.get("forecast", {})
    unavailable = check_unavailable_forecast(question, forecasts)
    if unavailable:
        return unavailable
    metrics = model_data or {}

    evidence = {
        "product_code": product_code,
        "forecast": forecasts,
        "champion": metrics.get("champion"),
        "challenger": metrics.get("challenger"),
        "champion_metrics": metrics.get("champion_metrics", {}),
        "challenger_metrics": metrics.get("challenger_metrics", {}),
        "registry_version": metrics.get("version"),
    }

    if not forecasts and not metrics:
        return "No stored forecast or model evaluation is available."

    client = genai.Client(
        vertexai=True,
        project=os.getenv("GOOGLE_CLOUD_PROJECT", "pharma-auto-forecasting"),
        location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
    )

    instruction = """
You are a pharmaceutical forecasting data assistant.

Answer ONLY using the supplied JSON evidence.
Do not use outside knowledge or internet information.
Do not invent, estimate, extrapolate, or calculate new forecasts.
Do not claim any evaluation or workflow was executed.
If a requested value is missing, say it is unavailable.
Never substitute a value from another month or medicine.
Keep answers concise.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=(
            f"USER QUESTION:\n{question}\n\n"
            f"VERIFIED STORED EVIDENCE:\n{json.dumps(evidence)}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=0,
        ),
    )

    answer = (response.text or "").strip()
    if not answer:
        return "Unable to answer from the available stored records."

    # Reject numeric claims not grounded in the supplied evidence.
    # This is a lightweight safeguard, not a complete semantic validator.
    numeric_tokens = re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", answer)
    evidence_numbers = set(
        re.findall(r"\d+(?:\.\d+)?", json.dumps(evidence))
    )

    for token in numeric_tokens:
        normalized = token.replace(",", "")
        if normalized not in evidence_numbers:
            return (
                "I couldn't verify the generated answer against "
                "the stored data. Please ask for a specific forecast "
                "month or model metric."
            )

    return answer
