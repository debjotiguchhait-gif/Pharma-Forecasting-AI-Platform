
import os
from pathlib import Path

from tools.monthly_ingestion import ingest_monthly_data
from tools.cloud_persistence import commit_dataset_to_gcs


def commit_cloud_monthly_data(incoming_path):
    # Validate without modifying historical data.
    preview = ingest_monthly_data(
        incoming_path=incoming_path,
        commit=False,
    )

    if preview["status"] == "ALREADY_INGESTED":
        return preview

    # Commit locally, using the existing ingestion logic.
    result = ingest_monthly_data(
        incoming_path=incoming_path,
        commit=True,
    )

    # Publish the updated dataset to Cloud Storage.
    dataset_path = Path(
        os.environ["PHARMA_DATA_PATH"]
    )

    snapshot = commit_dataset_to_gcs(dataset_path)

    result["cloud_snapshot"] = snapshot
    result["cloud_committed"] = True

    return result
