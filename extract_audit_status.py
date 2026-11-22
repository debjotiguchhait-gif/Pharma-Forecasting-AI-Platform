
import json
from pathlib import Path

import pandas as pd

AUDIT_FILE = Path("audit_log.jsonl")
TRACKER_FILE = Path("forecast_progress.csv")

# Load existing progress tracker
tracker = pd.read_csv(
    TRACKER_FILE,
    dtype={"product_code": str}
)

# Latest terminal audit record per medicine
latest = {}

with AUDIT_FILE.open("r", encoding="utf-8") as file:
    for line in file:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue

        if record.get("action") != "FORECAST_AUDIT":
            continue

        details = record.get("details") or {}
        code = details.get("product_code")

        if not code:
            continue

        timestamp = record.get("timestamp", "")

        if (
            code not in latest
            or timestamp >= latest[code]["timestamp"]
        ):
            latest[code] = {
                "timestamp": timestamp,
                "status": record.get("status"),
                "champion_model": details.get("champion_model"),
                "reason": details.get("reason"),
                "pending_months": details.get("pending_months", []),
                "forecast_generated": details.get(
                    "forecast_generated", False
                ),
            }

# Merge audit status with existing GCS forecast status
statuses = []
reasons = []
pending = []
audit_champions = []

for row in tracker.itertuples():
    audit = latest.get(row.product_code)

    if row.forecast_saved:
        status = "SUCCESS"
    elif audit:
        status = audit["status"] or "UNKNOWN"
    else:
        status = "NOT_PROCESSED"

    statuses.append(status)
    reasons.append(audit["reason"] if audit else None)
    pending.append(
        ",".join(map(str, audit["pending_months"] or []))
        if audit else ""
    )
    audit_champions.append(
        audit["champion_model"] if audit else None
    )

tracker["status"] = statuses
tracker["reason"] = reasons
tracker["pending_months"] = pending
tracker["audit_champion_model"] = audit_champions

tracker.to_csv(TRACKER_FILE, index=False)

print("\nFORECAST STATUS SUMMARY")
print("-----------------------")
print("Total medicines:", len(tracker))
print(tracker["status"].value_counts().to_string())
print("Tracker updated:", TRACKER_FILE.resolve())
