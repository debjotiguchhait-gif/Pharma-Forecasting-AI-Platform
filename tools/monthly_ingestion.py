
from pathlib import Path
import pandas as pd
import shutil
from uuid import uuid4
from datetime import datetime, timezone
from audit.logger import record_event
from agents.data_agent import assess_product_eligibility
import os
DATA_PATH = Path(
    os.environ.get(
        "PHARMA_DATA_PATH",
        "data/raw/nhs_monthly_prescribing.csv",
    )
)

REQUIRED_COLUMNS = [
    "month",
    "bnf_chemical_substance_code",
    "items",
]


def normalize_data(df):
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()

    df["bnf_chemical_substance_code"] = (
        df["bnf_chemical_substance_code"].astype("string")
    )

    df["month"] = pd.to_datetime(
        df["month"], errors="coerce"
    ).dt.to_period("M")

    df["items"] = pd.to_numeric(
        df["items"], errors="coerce"
    )

    if (
        df["month"].isna().any()
        or df["bnf_chemical_substance_code"].isna().any()
        or df["items"].isna().any()
        or (df["items"] < 0).any()
    ):
        raise ValueError("Invalid monthly prescribing observations")

    if df.duplicated(
        ["bnf_chemical_substance_code", "month"]
    ).any():
        raise ValueError("Duplicate product-month observations")

    return df


def ingest_monthly_data(incoming_path, commit=False):
    incoming = normalize_data(pd.read_csv(
        incoming_path,
        dtype={"bnf_chemical_substance_code": str},
    ))

    historical = normalize_data(pd.read_csv(
        DATA_PATH,
        dtype={"bnf_chemical_substance_code": str},
    ))

    keys = ["bnf_chemical_substance_code", "month"]

    existing = historical.set_index(keys)
    new = incoming.set_index(keys)

    overlap = existing.index.intersection(new.index)

    if len(overlap):
        existing_overlap = existing.loc[overlap, "items"]
        incoming_overlap = new.loc[overlap, "items"]

        identical = existing_overlap.equals(incoming_overlap)

        if identical and len(overlap) == len(new):
            return {
                "status": "ALREADY_INGESTED",
                "incoming_rows": len(incoming),
                "affected_products": [],
                "affected_product_count": 0,
                "commit": False,
            }

        raise ValueError(
            "Incoming data overlaps existing records. "
            "Partial duplicates or conflicting values are not permitted."
        )

    combined = pd.concat(
        [historical, incoming],
        ignore_index=True,
    ).sort_values(keys)

    affected_products = sorted(
        incoming["bnf_chemical_substance_code"]
        .unique()
        .tolist()
    )

    result = {
        "status": "VALIDATED",
        "incoming_rows": len(incoming),
        "affected_products": affected_products,
        "affected_product_count": len(affected_products),
        "commit": commit,
    }


    if commit:
        transaction_id = str(uuid4())

        snapshot_dir = Path("outputs/dataset_snapshots")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        snapshot_path = (
            snapshot_dir
            / f"nhs_prescribing_{timestamp}_{transaction_id}.csv"
        )

        temp_path = DATA_PATH.with_suffix(".tmp")

        result["transaction_id"] = transaction_id
        result["snapshot_path"] = str(snapshot_path)

        try:
            # Preserve the previous dataset before modification.
            shutil.copy2(DATA_PATH, snapshot_path)

            combined["month"] = combined["month"].astype(str)
            combined.to_csv(temp_path, index=False)

            # Replace only after the new dataset is fully written.
            temp_path.replace(DATA_PATH)

            result["status"] = "COMMITTED"

            record_event(
                user_id="system",
                action="MONTHLY_DATA_INGESTION",
                status="SUCCESS",
                details={
                    "transaction_id": transaction_id,
                    "incoming_rows": len(incoming),
                    "affected_products": affected_products,
                    "snapshot_path": str(snapshot_path),
                },
            )

        except Exception as exc:
            # Restore the previous dataset if commit fails.
            if snapshot_path.exists():
                shutil.copy2(snapshot_path, DATA_PATH)

            record_event(
                user_id="system",
                action="MONTHLY_DATA_INGESTION",
                status="FAILED",
                details={
                    "transaction_id": transaction_id,
                    "reason": str(exc),
                },
            )

            raise

        finally:
            if temp_path.exists():
                temp_path.unlink()


    return result
