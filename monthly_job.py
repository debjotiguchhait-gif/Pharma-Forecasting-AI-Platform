
"""Cloud Run entry point for monthly forecasting reassessment."""

import asyncio
import json
import os
from pathlib import Path

from google.cloud import storage

BUCKET = os.getenv(
    "PHARMA_GCS_BUCKET",
    "pharma-auto-forecasting-data",
)

DATA_PATH = "/tmp/pharma/data/nhs_monthly_prescribing.csv"

# Must be configured before importing forecasting modules.
os.environ["PHARMA_DATA_PATH"] = DATA_PATH


async def main():
    incoming_blob = os.environ.get("PHARMA_INCOMING_BLOB")

    if not incoming_blob:
        raise ValueError(
            "PHARMA_INCOMING_BLOB must specify a GCS CSV object."
        )

    client = storage.Client()
    bucket = client.bucket(BUCKET)

    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)

    bucket.blob(
        "raw/nhs_monthly_prescribing.csv"
    ).download_to_filename(DATA_PATH)

    incoming_path = "/tmp/pharma/incoming_monthly.csv"

    bucket.blob(incoming_blob).download_to_filename(
        incoming_path
    )

    # Import after configuring the dataset path.
    from tools.cloud_monthly_reassessment import (
        run_cloud_monthly_reassessment,
    )

    result = await run_cloud_monthly_reassessment(
        incoming_path
    )

    print(json.dumps(result, indent=2, default=str), flush=True)

    if result.get("status") not in {
        "COMPLETED",
        "SKIPPED",
    }:
        raise RuntimeError(
            f"Monthly reassessment did not complete: {result}"
        )


if __name__ == "__main__":
    asyncio.run(main())
