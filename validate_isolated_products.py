import json
from collections import Counter
from pathlib import Path

import pandas as pd

from tools.validate_imputation import validate_imputation
from tools.time_series import prepare_time_series

DATA_PATH = "data/raw/nhs_monthly_prescribing.csv"
OUTPUT_PATH = Path("outputs/imputation_validation.json")

records = [
    json.loads(line)
    for line in open("outputs/batch_forecasts.jsonl", encoding="utf-8")
    if line.strip()
]

records = [
    r for r in records
    if r.get("execution_mode") == "MCP_LANGGRAPH"
]

fingerprint = records[-1]["run_fingerprint"]

latest = {
    r["product_code"]: r
    for r in records
    if r.get("run_fingerprint") == fingerprint
}

results = []

for code, record in latest.items():
    if record.get("reason") != "Missing monthly observations require treatment":
        continue

    series = prepare_time_series(DATA_PATH, code)["series"]
    missing = series.isna()

    missing_count = int(missing.sum())
    longest_gap = int(
        missing.astype(int)
        .groupby((~missing).cumsum())
        .sum()
        .max()
    )

    if missing_count <= 3 and longest_gap <= 2:
        result = validate_imputation(code)
        results.append(result)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("PRODUCTS VALIDATED:", len(results))
print(
    "RECOMMENDED METHODS:",
    dict(Counter(
        r.get("recommended_method", r["status"])
        for r in results
    ))
)