
import io
import json
import sqlite3
import tempfile

import pandas as pd
from google.cloud import storage

BUCKET = "pharma-auto-forecasting-test"


def get_bucket():
    return storage.Client().bucket(BUCKET)


def get_products():
    bucket = get_bucket()
    return sorted(
        blob.name.split("/")[-1].removesuffix(".json")
        for blob in bucket.list_blobs(prefix="forecasts/")
        if blob.name.endswith(".json")
    )


def get_forecast(product_code):
    blob = get_bucket().blob(f"forecasts/{product_code}.json")
    if not blob.exists():
        return None
    return json.loads(blob.download_as_text())


def get_history(product_code):
    blob = get_bucket().blob("raw/nhs_monthly_prescribing.csv")
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))
    return df[
        df["bnf_chemical_substance_code"].astype(str) == product_code
    ].sort_values("month")



def get_model_performance(product_code):
    blob = get_bucket().blob("registry/model_registry.db")

    # Read SQLite database directly from GCS into memory
    db_bytes = blob.download_as_bytes()

    conn = sqlite3.connect(":memory:")
    try:
        conn.deserialize(db_bytes)

        row = conn.execute("""
            SELECT version, champion_model, challenger_model,
                   champion_metrics, challenger_metrics, created_at
            FROM model_registry
            WHERE product_code = ?
            ORDER BY version DESC, id DESC
            LIMIT 1
        """, (product_code,)).fetchone()

    finally:
        conn.close()

    if not row:
        return None

    return {
        "version": row[0],
        "champion": row[1],
        "challenger": row[2],
        "champion_metrics": json.loads(row[3]) if row[3] else {},
        "challenger_metrics": json.loads(row[4]) if row[4] else {},
        "created_at": row[5],
    }


def get_medicine_catalog():
    blob = get_bucket().blob("raw/nhs_monthly_prescribing.csv")
    df = pd.read_csv(io.BytesIO(blob.download_as_bytes()))

    code_col = "bnf_chemical_substance_code"
    name_col = "bnf_chemical_substance"

    available_codes = set(get_products())

    catalog = (
        df[[code_col, name_col]]
        .dropna(subset=[code_col])
        .drop_duplicates(subset=[code_col])
        .copy()
    )

    catalog[code_col] = catalog[code_col].astype(str)
    catalog = catalog[catalog[code_col].isin(available_codes)]

    catalog[name_col] = catalog[name_col].fillna("Unknown medicine")

    return {
        row[code_col]: f"{row[name_col]} | {row[code_col]}"
        for _, row in catalog.iterrows()
    }
