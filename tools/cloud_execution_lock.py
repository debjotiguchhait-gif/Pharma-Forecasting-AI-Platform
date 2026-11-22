
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage


BUCKET_NAME = os.environ.get(
    "PHARMA_GCS_BUCKET",
    "pharma-auto-forecasting-data",
)

LOCK_NAME = "locks/monthly_forecasting.lock"


def acquire_lock():
    client = storage.Client()
    blob = client.bucket(BUCKET_NAME).blob(LOCK_NAME)

    execution_id = uuid4().hex
    created_at = datetime.now(timezone.utc)

    # Cloud Run execution must finish within this period.
    expires_at = created_at + timedelta(hours=2)

    payload = {
        "execution_id": execution_id,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
    }

    import json

    try:
        blob.upload_from_string(
            json.dumps(payload),
            content_type="application/json",
            if_generation_match=0,
        )
    except PreconditionFailed:
        raise RuntimeError(
            "Another monthly forecasting execution "
            "already holds the cloud lock."
        )

    print(f"Acquired execution lock: {execution_id}")

    return execution_id, blob.generation


def release_lock(execution_id, generation):
    client = storage.Client()
    blob = client.bucket(BUCKET_NAME).blob(LOCK_NAME)

    blob.delete(if_generation_match=generation)

    print(f"Released execution lock: {execution_id}")
