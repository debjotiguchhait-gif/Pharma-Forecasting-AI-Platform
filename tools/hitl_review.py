
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

REVIEW_FILE = Path("outputs/anomaly_reviews.jsonl")


def record_review(
    product_code,
    month,
    decision,
    reviewer,
    justification,
    observed_value
):
    if decision not in ["APPROVED", "REJECTED"]:
        raise ValueError("Invalid review decision")

    if not reviewer or not reviewer.strip():
        raise ValueError("Reviewer is required")

    if not justification or not justification.strip():
        raise ValueError("Justification is required")

    REVIEW_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    review = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "product_code": str(product_code),
        "month": str(month),
        "decision": decision,
        "reviewer": reviewer,
        "justification": justification,
        "observation_hash": observation_hash(
            product_code, month, observed_value
        ),
    }

    with REVIEW_FILE.open(
        "a",
        encoding="utf-8"
    ) as file:
        file.write(json.dumps(review) + "\n")

    return review


def get_latest_review(product_code, month):
    if not REVIEW_FILE.exists():
        return None

    latest = None

    with REVIEW_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:
        for line in file:
            review = json.loads(line)

            if (
                review["product_code"] == str(product_code)
                and review["month"] == str(month)
            ):
                latest = review

    return latest

def observation_hash(product_code, month, value):
    payload = json.dumps(
        {
            "product_code": str(product_code),
            "month": str(month),
            "items": float(value)
        },
        sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()
