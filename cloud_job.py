"""Cloud Run entrypoint for Pharma Forecasting.

Default mode is read-only preview. Production commits are intentionally blocked
until the ingestion/checkpoint implementation is validated as crash-safe.
"""
import asyncio
import json
import os
from pathlib import Path

from google.cloud import storage

BUCKET_NAME = os.environ.get("PHARMA_GCS_BUCKET", "pharma-auto-forecasting-data")
HISTORICAL_PATH = "/tmp/pharma/data/nhs_monthly_prescribing.csv"
INCOMING_PATH = "/tmp/pharma/incoming/monthly.csv"


def download_file(blob_name: str, destination: str) -> None:
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    storage.Client().bucket(BUCKET_NAME).blob(blob_name).download_to_filename(
        str(destination_path)
    )
    print(f"Downloaded gs://{BUCKET_NAME}/{blob_name}")


async def main() -> None:
    # Set the path before importing project modules: some resolve it at import time.
    os.environ["PHARMA_DATA_PATH"] = HISTORICAL_PATH

    # Do not allow an unverified partial-commit workflow to mutate production data.
    if os.environ.get("PHARMA_COMMIT_ENABLED", "false").lower() == "true":
        raise RuntimeError(
            "Production commits are not yet safe: checkpoint creation, "
            "dataset publication, and state recovery require verification. "
            "Set PHARMA_COMMIT_ENABLED=false for a read-only preview."
        )

    from tools.cloud_persistence import restore_state
    from tools.monthly_reassessment import reassess_monthly_data

    download_file("raw/nhs_monthly_prescribing.csv", HISTORICAL_PATH)
    restore_state()

    incoming_blob = os.environ.get("PHARMA_INCOMING_BLOB")
    if not incoming_blob:
        print("No incoming monthly file; historical dataset and state loaded.")
        return

    download_file(incoming_blob, INCOMING_PATH)
    result = await reassess_monthly_data(
        incoming_path=INCOMING_PATH,
        commit=False,
    )
    print(json.dumps(result, indent=2, default=str))
    print("Preview complete; no cloud data or state was committed.")


if __name__ == "__main__":
    asyncio.run(main())
