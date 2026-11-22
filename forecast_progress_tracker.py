
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from google.cloud import storage

BUCKET = "pharma-auto-forecasting-data"
DATA_FILE = "nhs_monthly_prescribing.csv"
OUTPUT = Path("forecast_progress.csv")

client = storage.Client()
bucket = client.bucket(BUCKET)

df = pd.read_csv(
    DATA_FILE,
    dtype={"bnf_chemical_substance_code": str}
)
df["month"] = pd.to_datetime(
    df["month"], format="%Y-%m", errors="coerce"
)
df["items"] = pd.to_numeric(df["items"], errors="coerce")
df = df.dropna(
    subset=["month", "bnf_chemical_substance_code", "items"]
)
df = df[df["items"] >= 0]

counts = df.groupby(
    "bnf_chemical_substance_code"
)["month"].nunique()

codes = sorted(counts[counts >= 36].index.tolist())

# Read saved forecast files directly from GCS.
saved = {}

for blob in client.list_blobs(BUCKET, prefix="forecasts/"):
    if not blob.name.endswith(".json"):
        continue

    try:
        payload = json.loads(blob.download_as_text())
        code = payload.get("product_code")
        forecast = payload.get("forecast_result") or {}

        if (
            code in codes
            and forecast.get("status") == "SUCCESS"
        ):
            saved[code] = {
                "champion_model": payload.get("champion_model"),
                "forecast_saved": True,
            }
    except Exception as exc:
        print(f"Could not read {blob.name}: {exc}")

# Build a complete 498-medicine tracker.
rows = []

for offset, code in enumerate(codes):
    result = saved.get(code, {})

    rows.append({
        "batch_offset": offset,
        "product_code": code,
        "forecast_saved": result.get("forecast_saved", False),
        "champion_model": result.get("champion_model"),
        "status": (
            "SUCCESS"
            if result.get("forecast_saved")
            else "NOT_SAVED"
        ),
    })

tracker = pd.DataFrame(rows)
tracker.to_csv(OUTPUT, index=False)

print("\nFORECASTING PROGRESS")
print("--------------------")
print("Eligible medicines:", len(tracker))
print("Saved forecasts:", int(tracker["forecast_saved"].sum()))
print("Without saved forecast:", int((~tracker["forecast_saved"]).sum()))
print("Status counts:", dict(Counter(tracker["status"])))
print("Tracker saved to:", OUTPUT.resolve())
