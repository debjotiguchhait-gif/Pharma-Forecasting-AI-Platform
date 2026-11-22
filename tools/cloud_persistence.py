
import os
from pathlib import Path

from google.cloud import storage
from datetime import datetime, timezone
from uuid import uuid4

BUCKET_NAME = os.environ.get(
    "PHARMA_GCS_BUCKET",
    "pharma-auto-forecasting-data",
)

PERSISTENT_FILES = {
    "outputs/model_registry.db": "registry/model_registry.db",
    "outputs/audit_log.jsonl": "audit/audit_log.jsonl",
    "outputs/batch_forecasts.jsonl": "forecasts/batch_forecasts.jsonl",
}


def restore_state():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for local_name, cloud_name in PERSISTENT_FILES.items():
        blob = bucket.blob(cloud_name)

        if not blob.exists():
            print(f"No previous state: {cloud_name}")
            continue

        local_path = Path(local_name)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        blob.download_to_filename(str(local_path))
        print(f"Restored: {cloud_name}")


def save_state():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    for local_name, cloud_name in PERSISTENT_FILES.items():
        local_path = Path(local_name)

        if not local_path.exists():
            continue

        bucket.blob(cloud_name).upload_from_filename(
            str(local_path)
        )

        print(f"Saved: {cloud_name}")



def commit_dataset_to_gcs(local_dataset_path):
    """
    Upload an updated dataset to GCS.

    Creates a cloud snapshot before replacing the
    current dataset. Uses a generation precondition
    to prevent overwriting a newer cloud version.
    """
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    dataset_blob = bucket.blob(
        "raw/nhs_monthly_prescribing.csv"
    )
    dataset_blob.reload()

    original_generation = dataset_blob.generation

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    snapshot_name = (
        f"snapshots/nhs_prescribing_"
        f"{timestamp}_{uuid4().hex}.csv"
    )

    # Snapshot the existing cloud dataset.
    bucket.copy_blob(
        dataset_blob,
        bucket,
        new_name=snapshot_name,
        if_source_generation_match=original_generation,
    )

    # Replace only if the original generation is unchanged.
    dataset_blob.upload_from_filename(
        str(local_dataset_path),
        if_generation_match=original_generation,
    )

    print(f"Dataset snapshot: {snapshot_name}")
    print("Updated historical dataset saved to GCS")

    return snapshot_name
