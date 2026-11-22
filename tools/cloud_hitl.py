"""Immutable HITL reviews in GCS; first decision wins."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from io import StringIO

import pandas as pd
from google.api_core.exceptions import PreconditionFailed, Conflict
from google.cloud import storage
from tools.hitl_review import observation_hash

BUCKET = os.getenv("PHARMA_GCS_BUCKET", "pharma-auto-forecasting-data")
REVIEW_PREFIX = "hitl/reviews/"
LOCAL_REVIEW_FILE = Path("outputs/anomaly_reviews.jsonl")


def load_latest_reviews():
    """Read historical migrated records and new deterministic decision objects."""
    client = storage.Client()
    latest = {}
    for blob in client.list_blobs(BUCKET, prefix=REVIEW_PREFIX):
        if not blob.name.endswith(".json"):
            continue
        record = json.loads(blob.download_as_text())
        if record.get("decision") not in {"APPROVED", "REJECTED"}:
            continue
        key = (str(record["product_code"]), str(record["month"]))
        if key not in latest or record["timestamp"] > latest[key]["timestamp"]:
            latest[key] = record
    return latest


def sync_reviews_for_forecasting():
    """Rebuild the local review file used by the existing forecasting pipeline."""
    records = sorted(load_latest_reviews().values(), key=lambda r: (r["timestamp"], r["product_code"], r["month"]))
    LOCAL_REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = LOCAL_REVIEW_FILE.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(LOCAL_REVIEW_FILE)
    return len(records)


def submit_review_once(product_code, month, decision, reviewer, justification, observed_value):
    """Validate latest GCS observation and atomically create one decision."""
    code, month = str(product_code), str(month)
    if decision not in {"APPROVED", "REJECTED"}:
        raise ValueError("Invalid decision")
    if not reviewer or not reviewer.strip():
        raise ValueError("Verified reviewer required")
    if not justification or not justification.strip():
        raise ValueError("Scientific justification required")
    if len(justification.strip()) > 5000:
        raise ValueError("Justification too long")
    if (code, month) in load_latest_reviews():
        raise ValueError("Already reviewed. Refresh dashboard.")

    client = storage.Client()
    bucket = client.bucket(BUCKET)
    source = bucket.blob("raw/nhs_monthly_prescribing.csv")
    df = pd.read_csv(StringIO(source.download_as_text()), dtype={"bnf_chemical_substance_code": str})
    matches = df.loc[(df["bnf_chemical_substance_code"] == code) & (df["month"].astype(str) == month)]
    if len(matches) != 1:
        raise ValueError("Missing or duplicated observation")
    current = pd.to_numeric(matches.iloc[0]["items"], errors="coerce")
    if pd.isna(current) or current < 0:
        raise ValueError("Invalid current observation")
    if observation_hash(code, month, current) != observation_hash(code, month, observed_value):
        raise ValueError("Observation changed; refresh dashboard")

    record = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "product_code": code, "month": month, "decision": decision,
              "reviewer": reviewer.strip(), "justification": justification.strip(),
              "observation_hash": observation_hash(code, month, current)}
    # Atomic create-only for all NEW decisions; historical migrated records are
    # protected by the existence check above.
    blob = bucket.blob(f"{REVIEW_PREFIX}decisions/{code}/{month}.json")
    try:
        blob.upload_from_string(json.dumps(record), content_type="application/json", if_generation_match=0)
    except (PreconditionFailed, Conflict) as exc:
        raise ValueError("Another reviewer already decided. Refresh dashboard.") from exc
    return record
