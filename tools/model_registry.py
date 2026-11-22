
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import inspect
import pandas as pd

import os

def get_registry_path():
    return Path(
        os.environ.get(
            "PHARMA_REGISTRY_PATH",
            "outputs/model_registry.db",
        )
    )
BATCH_PATH = Path("outputs/batch_forecasts.jsonl")


def connect_registry():
    registry_path = get_registry_path()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(registry_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_code TEXT NOT NULL,
            version INTEGER NOT NULL,
            champion_model TEXT NOT NULL,
            challenger_model TEXT,
            champion_metrics TEXT,
            challenger_metrics TEXT,
            run_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(product_code, version)
        )
    """)

    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_registry_run
        ON model_registry(product_code, run_fingerprint)
    """)
    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(model_registry)"
        )
    }

    if "evaluation_fingerprint" not in columns:
        conn.execute("""
            ALTER TABLE model_registry
            ADD COLUMN evaluation_fingerprint TEXT
        """)

    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_unique_product_evaluation
        ON model_registry(
            product_code,
            evaluation_fingerprint
        )
        WHERE evaluation_fingerprint IS NOT NULL
    """)
    conn.commit()
    return conn


def register_model(record, conn):
    product_code = record["product_code"]
    fingerprint = record["run_fingerprint"]

    existing = conn.execute(
        """
        SELECT id FROM model_registry
        WHERE product_code = ? AND run_fingerprint = ?
        """,
        (product_code, fingerprint),
    ).fetchone()

    if existing:
        return False

    next_version = conn.execute(
        """
        SELECT COALESCE(MAX(version), 0) + 1
        FROM model_registry
        WHERE product_code = ?
        """,
        (product_code,),
    ).fetchone()[0]

    conn.execute(
        """
        INSERT INTO model_registry (
            product_code, version, champion_model,
            challenger_model, champion_metrics,
            challenger_metrics, run_fingerprint, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            product_code,
            next_version,
            record["champion_model"],
            record.get("challenger_model"),
            json.dumps(record.get("champion_metrics")),
            json.dumps(record.get("challenger_metrics")),
            fingerprint,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    return True


def import_batch():
    if not BATCH_PATH.exists():
        raise FileNotFoundError(BATCH_PATH)

    latest = {}

    with BATCH_PATH.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record.get("status") == "SUCCESS"
                and record.get("champion_model")
                and record.get("run_fingerprint")
                and record.get("product_code")
            ):
                key = (
                    record["product_code"],
                    record["run_fingerprint"],
                )
                latest[key] = record

    conn = connect_registry()
    inserted = 0

    try:
        for record in latest.values():
            if register_model(record, conn):
                inserted += 1

        conn.commit()
    finally:
        conn.close()

    return {
        "successful_run_records": len(latest),
        "new_registry_entries": inserted,
    }


def get_latest_model(product_code):
    conn = connect_registry()

    try:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT * FROM model_registry
            WHERE product_code = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (str(product_code),),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)

        for field in ("champion_metrics", "challenger_metrics"):
            result[field] = (
                json.loads(result[field])
                if result[field] is not None
                else None
            )

        return result
    finally:
        conn.close()


def calculate_evaluation_fingerprint(
    series,
    product_code,
    horizon,
    min_train,
    candidate_models,
    approved_months=None,
    excluded_months=None,
):
    """Fingerprint model evaluation inputs and relevant code."""

    from tools import (
        time_series,
        backtesting,
        statistical_models,
    )
    from agents import ml_agent, evaluation_agent

    modules = (
        time_series,
        backtesting,
        statistical_models,
        ml_agent,
        evaluation_agent,
    )

    code_hashes = {}

    for module in modules:
        source = inspect.getsource(module)
        code_hashes[module.__name__] = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()

    observations = [
        {
            "month": str(month),
            "value": (
                None if pd.isna(value)
                else float(value)
            ),
        }
        for month, value in series.items()
    ]

    payload = {
        "product_code": str(product_code),
        "observations": observations,
        "horizon": horizon,
        "min_train": min_train,
        "candidate_models": list(candidate_models),
        "approved_months": sorted(approved_months or []),
        "excluded_months": sorted(excluded_months or []),
        "code_hashes": code_hashes,
        "fingerprint_schema": 1,
    }

    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()

def get_model_by_fingerprint(product_code, evaluation_fingerprint):
    """Find a model evaluated with exactly the same inputs."""
    conn = connect_registry()

    try:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT *
            FROM model_registry
            WHERE product_code = ?
              AND evaluation_fingerprint = ?
            ORDER BY version DESC
            LIMIT 1
            """,
            (str(product_code), evaluation_fingerprint),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)

        for field in ("champion_metrics", "challenger_metrics"):
            if result[field] is not None:
                result[field] = json.loads(result[field])

        return result
    finally:
        conn.close()

if __name__ == "__main__":
    print(import_batch())
