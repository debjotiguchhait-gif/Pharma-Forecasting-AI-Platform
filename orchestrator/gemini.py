
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT", "pharma-auto-forecasting"
)
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    project=PROJECT_ID,
    location=LOCATION,
    vertexai=True,
    max_retries=2,
)

def interpret_data(request, validation_result):
    prompt = f"""
You are the orchestration assistant for a pharmaceutical
forecasting platform.

User request:
{request}

Verified Data Agent result:
{validation_result}

Instructions:
- Use only the supplied verified results.
- Explain data quality and forecasting readiness.
- Distinguish provisional eligibility from final approval.
- Mention that missing months may reflect top-500 coverage.
- Do not invent metrics, train models or modify data.
- Keep the response concise.
"""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.text
